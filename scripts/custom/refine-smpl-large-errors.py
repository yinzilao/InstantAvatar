import numpy as np
import smplx
from tqdm import tqdm
import os
import glob
import sys
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    BlendParams,
    SoftSilhouetteShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_euler_angles
import trimesh
from typing import List

from utils.schp_utils import ModelType, get_hair_hat_mask, get_face_mask, get_schp_segmentation, get_schp_segmentation_batch

DEVICE = "cuda"
SHAPE_LR = 10e-2
POSE_LR = 5e-3
NO_HEAD_SILHOUETTE = False
USE_HAIR_HAT_FACE_WEIGHT = True

BATCHSIZE = 12  # Conservative batch size, increase if memory allows
def scale_gradients(parameters, clip_value=1.0):
    """
    Scale gradients to balance different parameter groups
    Args:
        parameters: list of parameter groups to scale
        clip_value: maximum gradient norm
    """
    # Get gradient statistics for each parameter group
    grad_norms = []
    for param_group in parameters:
        group_grads = []
        for p in param_group:
            if p.grad is not None:
                # Clip extremely large gradients
                torch.nn.utils.clip_grad_norm_(p, clip_value)
                group_grads.append(p.grad.norm())
        if group_grads:
            grad_norms.append(torch.stack(group_grads).mean())
    
    if not grad_norms:  # No gradients to scale
        return
    
    # Calculate mean norm across all groups
    mean_norm = torch.stack(grad_norms).mean()
    
    # Scale each parameter group
    for param_group, group_norm in zip(parameters, grad_norms):
        scale = mean_norm / (group_norm + 1e-8)  # Avoid division by zero
        scale = torch.clamp(scale, 0, clip_value)  # Prevent extreme scaling
        
        for p in param_group:
            if p.grad is not None:
                p.grad.mul_(scale)

def optimize(optimizer_shape, optimizer_pose, closure, params, batch_start, batch_end, max_iter=100):
    pbar = tqdm(range(max_iter))
    prev_loss = float('inf')
    patience = 3
    patience_counter = 0
    min_improvement = 1e-6

    scheduler_shape = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_shape, mode='min', factor=0.5, patience=patience)
    scheduler_pose = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_pose, mode='min', factor=0.5, patience=patience)
    
    for i in pbar:
        # First optimize shape
        optimizer_shape.zero_grad()
        loss_shape = closure(shape_only=True)
        
        # Debug info
        print(f"Shape optimization - loss_shape.requires_grad: {loss_shape.requires_grad}")
        print(f"params['betas'].requires_grad: {params['betas'].requires_grad}")
        
        # if not loss_shape.requires_grad:
        #     raise ValueError("Loss shape should require grad but it doesn't! "
        #                      f"loss_shape.requires_grad: {loss_shape.requires_grad}. "
        #                      "Check computational graph.")
            
        loss_shape.backward()
        scale_gradients(
            [params['betas']], # betas shared across all frames because body shape doesn't change
            clip_value=5.0
        ) 
        optimizer_shape.step()

        # Then optimize pose
        optimizer_pose.zero_grad()
        loss_pose = closure(shape_only=False)
        
        # Debug info
        print(f"Pose optimization - loss requires grad: {loss_pose.requires_grad}")
        print(f"Body pose requires grad: {params['body_pose'].requires_grad}")
        
        if not loss_pose.requires_grad:
            raise ValueError(f"Loss pose should require grad but it doesn't! "
                             f"loss_pose.requires_grad: {loss_pose.requires_grad}. "
                             "Check computational graph.")
            
        loss_pose.backward()
        scale_gradients(
            [params['body_pose'][batch_start:batch_end], 
              params['global_orient'][batch_start:batch_end],
              params['transl'][batch_start:batch_end]],
            clip_value=5.0
        )
        optimizer_pose.step()

        current_loss = (loss_shape + loss_pose).detach().cpu().numpy()

        # Update learning rates
        scheduler_shape.step(current_loss)
        scheduler_pose.step(current_loss)
        
        # Early stopping with relative improvement check
        improvement = (prev_loss - current_loss) / (prev_loss + 1e-10)  # Avoid division by zero
        if abs(improvement) < min_improvement:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at iteration {i}")
                break
        else:
            patience_counter = 0
            
        prev_loss = current_loss
        pbar.set_postfix_str(f"loss: {current_loss:.6f}")

        # Print gradient norms for debugging
        if i % 20 == 0:
            with torch.no_grad():
                beta_grad_norm = params['betas'].grad.norm().item() if params['betas'].grad is not None else 0
                pose_grad_norm = params['body_pose'].grad[batch_start:batch_end].norm().item() if params['body_pose'].grad is not None else 0
                print(f"\nGradient norms:")
                print(f"Beta gradients: {beta_grad_norm:.6f}")
                print(f"Pose gradients: {pose_grad_norm:.6f}")
        

def project(projection_matrices, keypoints_3d):
    p = torch.einsum("ij,mnj->mni", projection_matrices[:3, :3], keypoints_3d) + projection_matrices[:3, 3]
    p = p[..., :2] / p[..., 2:3]
    return p


def build_renderer(camera, IMG_SIZE):
    K = camera["intrinsic"]
    K = torch.from_numpy(K).float().to(DEVICE)

    R = torch.eye(3, device=DEVICE, dtype=torch.float32)[None]
    R[:, 0] *= -1
    R[:, 1] *= -1
    t = torch.zeros(1, 3, device=DEVICE, dtype=torch.float32)


    cameras = PerspectiveCameras(
        focal_length=K[None, [0, 1], [0, 1]],
        principal_point=K[None, [0, 1], [2, 2]],
        R=R,
        T=t,
        image_size=[IMG_SIZE],
        in_ndc=False,
        device=DEVICE,
    )
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    raster_settings = RasterizationSettings(
        image_size=IMG_SIZE,
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=100,
        bin_size=0,  # Set to 0 to use naive rasterization
        max_faces_per_bin=50000  # Increase this value
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(
            blend_params=blend_params
        )
    )
    return renderer

class BODY25JointMapper:
    SMPL_TO_BODY25 = [
        24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34
    ]

    def __init__(self):
        self.mapping = self.SMPL_TO_BODY25

    def __call__(self, smpl_output, *args, **kwargs):
        return smpl_output.joints[:, self.mapping]


HEATMAP_THRES = 0.30
PAF_THRES = 0.05
PAF_RATIO_THRES = 0.95
NUM_SAMPLE = 10
MIN_POSE_JOINT_COUNT = 4
MIN_POSE_LIMB_SCORE = 0.4
NUM_JOINTS = 25
BODY25_POSE_INDEX = [(0, 1), (14, 15), (22, 23), (16, 17), (18, 19), (24, 25),
                     (26, 27), (6, 7), (2, 3), (4, 5), (8, 9), (10, 11),
                     (12, 13), (30, 31), (32, 33), (36, 37), (34, 35),
                     (38, 39), (20, 21), (28, 29), (40, 41), (42, 43),
                     (44, 45), (46, 47), (48, 49), (50, 51)]
BODY25_PART_PAIRS = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
                     (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
                     (1, 0), (0, 15), (15, 17), (0, 16), (16, 18), (2, 17),
                     (5, 18), (14, 19), (19, 20), (14, 21), (11, 22), (22, 23),
                     (11, 24)]

joints_name = [
    "Nose",         # 0  => 24 (SMPL)
    "Neck",         # 1  => 12
    "RShoulder",    # 2  => 17
    "RElbow",       # 3  => 19
    "RWrist",       # 4  => 21
    "LShoulder",    # 5  => 16
    "LElbow",       # 6  => 18
    "LWrist",       # 7  => 20
    "MidHip",       # 8  => 0
    "RHip",         # 9  => 2
    "RKnee",        # 10 => 5
    "RAnkle",       # 11 => 8
    "LHip",         # 12 => 1
    "LKnee",        # 13 => 4
    "LAnkle",       # 14 => 7
    "REye",         # 15 => 25
    "LEye",         # 16 => 26
    "REar",         # 17 => 27
    "LEar",         # 18 => 28
    "LBigToe",      # 19 => 29
    "LSmallToe",    # 20 => 30
    "LHeel",        # 21 => 31
    "RBigToe",      # 22 => 32
    "RSmallToe",    # 23 => 33
    "RHeel",        # 24 => 34
]

HEAD_KEYPOINT_INDICES = [0, 15, 16, 17, 18]  # Nose, Eyes, Ears

SELECT_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 
                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


import cv2
def draw_detect(frame: np.ndarray, poses2d: np.ndarray, color=(255, 255, 0)):
    for person in poses2d:
        # draw parts
        for (i, j) in BODY25_PART_PAIRS:
            if person.shape[-1] > 2 and min(person[[i, j], 2]) < 1e-3:
                continue
            frame = cv2.line(frame, tuple(person[i, :2].astype(int)),
                             tuple(person[j, :2].astype(int)), (color), 1)

        # draw joints
        for joint in person:
            if len(joint) > 2 and joint[-1] == 0:
                continue
            pos = joint[:2].astype(int)
            frame = cv2.circle(frame, tuple(pos), 2, (color), 2, cv2.FILLED)
    return frame

def get_hair_hat_face_weight_mask(
        seg_map: np.ndarray, 
        model_type=ModelType.ATR, 
        weights: List[float] = [0.0, 0.5, 1.0]
    ):
    """Get weight mask for hair and hat regions"""
    hair_hat_mask = get_hair_hat_mask(seg_map, model_type)
    face_mask = get_face_mask(seg_map, model_type)
    # create weight mask (0.0 for hair and hat, 1.0 for other regions, 0.5 for face)
    hair_hat_face_weight_mask = np.ones_like(hair_hat_mask) * weights[2]
    hair_hat_face_weight_mask[hair_hat_mask > 0] = weights[0]
    hair_hat_face_weight_mask[face_mask > 0] = weights[1]
    return hair_hat_face_weight_mask

def save_mesh_and_measurements(body_model, params, root_dir):
    """Generate and save SMPL mesh and body measurements.
    
    Args:
        body_model: SMPL model instance
        params: Dict of SMPL parameters (betas, pose, etc.)
        root_dir: Output directory path
    
    Returns:
        dict: Containing measurements (shoulder_width, etc.)
    """
    # Generate mesh
    smpl_output = body_model(**params)
    vertices = smpl_output.vertices.detach().cpu().numpy()
    faces = body_model.faces

    # Save mesh
    mesh = trimesh.Trimesh(vertices[0], faces)
    os.makedirs(f"{root_dir}/refine-smpl", exist_ok=True)
    mesh.export(f"{root_dir}/refine-smpl/body_mesh.obj")

    # # Calculate measurements TBA
    # LEFT_SHOULDER_IDX = 2802
    # RIGHT_SHOULDER_IDX = 6262
    
    # left_shoulder = vertices[0, LEFT_SHOULDER_IDX]
    # right_shoulder = vertices[0, RIGHT_SHOULDER_IDX]
    # shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
    
    # print(f"Shoulder width: {shoulder_width*100:.1f} cm")

    # # Save measurements
    # measurements = {
    #     'shoulder_width': shoulder_width,
    #     'shoulder_points': np.stack([left_shoulder, right_shoulder])
    # }
    # np.savez(f"{root_dir}/refine-smpl/measurements.npz", **measurements)
    
    # return measurements

def save_posed_mesh(body_model, params, output_path, pose_type='zero'):
    """Save SMPL mesh in A-pose or T-pose"""
    # Keep optimized shape parameters
    new_params = {
        'betas': params['betas'],
        'global_orient': torch.zeros_like(params['global_orient']),  # Reset global rotation
        'transl': torch.zeros_like(params['transl'])  # Reset translation
    }

    body_pose = torch.zeros_like(params['body_pose'][0]) # default zero pose
    if pose_type.lower() == 't':
        # Set shoulder joints to 90 degrees (π/2 radians) for T-pose
        body_pose[16*3 + 1] = np.pi/2  # Left shoulder Y-axis
        body_pose[17*3 + 1] = -np.pi/2  # Right shoulder Y-axis
    elif pose_type.lower() == 'a':
        # Set shoulder joints to 45 degrees (π/4 radians) for A-pose
        body_pose[16*3 + 1] = np.pi/4  # Left shoulder Y-axis
        body_pose[17*3 + 1] = -np.pi/4  # Right shoulder Y-axis

    new_params['body_pose'] = body_pose.unsqueeze(0)
        
    smpl_output = body_model(**new_params)
    vertices = smpl_output.vertices.detach().cpu().numpy()
    faces = body_model.faces

    # save mesh
    mesh = trimesh.Trimesh(vertices[0], faces)
    mesh.export(output_path)

def temporal_smoothness_loss(vertices, window_size=3, loss_downgrade=1.0):
    """Compute smoothness loss over a sliding window"""
    total_loss = 0
    batch_size = vertices.shape[0]
    
    for i in range(window_size):
        if i == 0:
            continue
        # Compare frame t with frame t-i
        diff = vertices[i:] - vertices[:-i]

        # Clip extremely large differences to prevent explosion
        diff = torch.clamp(diff, min=-100.0, max=100.0)
        print(f"WARNING: frame diff clipped to: {diff}")

        # L1 loss for more robustness
        loss = diff.abs().mean(dim=-1)  # Mean across xyz dimensions
        loss = loss.mean()  # Mean across vertices

        # Add small epsilon to prevent division by zero
        weight = (1.0 / (i + 1e-6))

        # Weight decreases with temporal distance
        scaled_loss = loss * weight * loss_downgrade #TODO: check whether we need * 0.01 (grad clif) for handling large errors
        print(f"scaled loss: {scaled_loss}, loss: {loss}, weight: {weight}, loss_downgrade: {loss_downgrade}")

        if torch.isnan(scaled_loss):
            print(f"Warning: NaN detected in frame {i}")
            print(f"diff stats - min: {diff.min()}, max: {diff.max()}, mean: {diff.mean()}")
            continue

        total_loss += scaled_loss

    return total_loss / window_size

def pose_smoothness_loss(body_pose, window_size=3):
    """Compute angle-based smoothness loss for body poses"""
    total_loss = 0
    batch_size = body_pose.shape[0]
    
    # Convert to rotation matrices
    pose_mat = axis_angle_to_matrix(body_pose.reshape(-1, 3)).reshape(batch_size, 23, 3, 3)
    
    for i in range(window_size):
        if i == 0:
            continue
        # Compare rotations between frames
        rot_diff = torch.matmul(
            pose_mat[i:], 
            pose_mat[:-i].transpose(-1, -2)
        )
        
        # Convert to geodesic distance (angle difference)
        theta = torch.acos(torch.clamp(
            (torch.diagonal(rot_diff, dim1=-2, dim2=-1).sum(-1) - 1) / 2,
            -1 + 1e-7, 
            1 - 1e-7
        ))
        
        # Weight by temporal distance
        loss = theta.mean() * (1.0 / i)
        total_loss += loss
    
    return total_loss / window_size

def pose_prior_loss(body_pose):
    """Penalize extreme pose deviations"""
    # Convert to euler angles for easier interpretation
    pose_mat = axis_angle_to_matrix(body_pose.reshape(-1, 3))
    euler_angles = matrix_to_euler_angles(pose_mat, "XYZ")
    
    # Penalize large deviations from neutral pose
    threshold = torch.pi / 4  # 45 degrees
    excess = (euler_angles.abs() - threshold).clamp(min=0)
    return excess.mean()

def get_joint_limits_and_weights():
    """
    Define anatomical joint limits and their weights in the loss function
    All angles in radians
    
    Returns:
        dict: Joint limits and weights
        {joint_name: {'limits': {'x': (min, max), 'y': (min, max), 'z': (min, max)}, 
                     'weight': float}}
    """
    return {
        # Torso
        'MidHip': {  # Root joint
            'limits': {
                'x': (-0.4*np.pi, 0.4*np.pi),  # forward/backward tilt
                'y': (-0.3*np.pi, 0.3*np.pi),  # side bend
                'z': (-0.3*np.pi, 0.3*np.pi),  # rotation
            },
            'weight': 2.0  # Important for overall pose
        },
        
        # Spine and Head
        'Neck': {
            'limits': {
                'x': (-0.01*np.pi, 0.01*np.pi),  # Very limited nodding (was -0.5*np.pi, 0.5*np.pi)
                'y': (-0.01*np.pi, 0.01*np.pi),  # Very limited side bending (was -0.3*np.pi, 0.3*np.pi)
                'z': (-0.01*np.pi, 0.01*np.pi),  # Very limited rotation (was -0.7*np.pi, 0.7*np.pi)
            },
            'weight': 5.0  # Increased weight (was 2.5)
        },
        'Nose': {  # Head joint
            'limits': {
                'x': (-0.01*np.pi, 0.01*np.pi),  # Almost no tilt (was -0.3*np.pi, 0.3*np.pi)
                'y': (-0.01*np.pi, 0.01*np.pi),  # Almost no side tilt (was -0.2*np.pi, 0.2*np.pi)
                'z': (-0.01*np.pi, 0.01*np.pi),  # Almost no rotation (was -0.4*np.pi, 0.4*np.pi)
            },
            'weight': 5.0  # Increased weight (was 3.0)
        },
        
        # Arms
        'LShoulder': {
            'limits': {
                'x': (-np.pi, np.pi),      # full rotation
                'y': (-0.5*np.pi, 0.5*np.pi),  # up/down
                'z': (-0.5*np.pi, 0.5*np.pi),  # forward/backward
            },
            'weight': 1.5
        },
        'RShoulder': {
            'limits': {
                'x': (-np.pi, np.pi),
                'y': (-0.5*np.pi, 0.5*np.pi),
                'z': (-0.5*np.pi, 0.5*np.pi),
            },
            'weight': 1.5
        },
        'LElbow': {
            'limits': {
                'x': (-0.1*np.pi, 0.1*np.pi),  # very limited sideways
                'y': (-0.1*np.pi, 0.1*np.pi),  # very limited rotation
                'z': (0, 2.7),  # main flexion (positive for left arm)
            },
            'weight': 2.0
        },
        'RElbow': {
            'limits': {
                'x': (-0.1*np.pi, 0.1*np.pi),
                'y': (-0.1*np.pi, 0.1*np.pi),
                'z': (-2.7, 0),  # negative for right arm
            },
            'weight': 2.0
        },
        'LWrist': {
            'limits': {
                'x': (-0.5*np.pi, 0.5*np.pi),  # flexion/extension
                'y': (-0.3*np.pi, 0.3*np.pi),  # ulnar/radial deviation
                'z': (-0.5*np.pi, 0.5*np.pi),  # pronation/supination
            },
            'weight': 1.0
        },
        'RWrist': {
            'limits': {
                'x': (-0.5*np.pi, 0.5*np.pi),
                'y': (-0.3*np.pi, 0.3*np.pi),
                'z': (-0.5*np.pi, 0.5*np.pi),
            },
            'weight': 1.0
        },
        
        # Legs
        'LHip': {
            'limits': {
                'x': (-0.7*np.pi, 0.7*np.pi),  # forward/backward
                'y': (-0.3*np.pi, 0.3*np.pi),  # abduction/adduction
                'z': (-0.3*np.pi, 0.3*np.pi),  # rotation
            },
            'weight': 2.0
        },
        'RHip': {
            'limits': {
                'x': (-0.7*np.pi, 0.7*np.pi),
                'y': (-0.3*np.pi, 0.3*np.pi),
                'z': (-0.3*np.pi, 0.3*np.pi),
            },
            'weight': 2.0
        },
        'LKnee': {
            'limits': {
                'x': (0, 0.7*np.pi),  # flexion only
                'y': (-0.1*np.pi, 0.1*np.pi),  # very limited sideways
                'z': (-0.1*np.pi, 0.1*np.pi),  # very limited rotation
            },
            'weight': 2.5
        },
        'RKnee': {
            'limits': {
                'x': (0, 0.7*np.pi),
                'y': (-0.1*np.pi, 0.1*np.pi),
                'z': (-0.1*np.pi, 0.1*np.pi),
            },
            'weight': 2.5
        },
        'LAnkle': {
            'limits': {
                'x': (-0.5*np.pi, 0.3*np.pi),  # plantar/dorsiflexion
                'y': (-0.3*np.pi, 0.3*np.pi),  # inversion/eversion
                'z': (-0.3*np.pi, 0.3*np.pi),  # rotation
            },
            'weight': 1.5
        },
        'RAnkle': {
            'limits': {
                'x': (-0.5*np.pi, 0.3*np.pi),
                'y': (-0.3*np.pi, 0.3*np.pi),
                'z': (-0.3*np.pi, 0.3*np.pi),
            },
            'weight': 1.5
        },
        
        # Feet
        'LBigToe': {
            'limits': {
                'x': (-0.3*np.pi, 0.3*np.pi),
                'y': (-0.1*np.pi, 0.1*np.pi),
                'z': (-0.1*np.pi, 0.1*np.pi),
            },
            'weight': 1.0
        },
        'RBigToe': {
            'limits': {
                'x': (-0.3*np.pi, 0.3*np.pi),
                'y': (-0.1*np.pi, 0.1*np.pi),
                'z': (-0.1*np.pi, 0.1*np.pi),
            },
            'weight': 1.0
        },
        'LSmallToe': {
            'limits': {
                'x': (-0.3*np.pi, 0.3*np.pi),
                'y': (-0.1*np.pi, 0.1*np.pi),
                'z': (-0.1*np.pi, 0.1*np.pi),
            },
            'weight': 1.0
        },
        'RSmallToe': {
            'limits': {
                'x': (-0.3*np.pi, 0.3*np.pi),
                'y': (-0.1*np.pi, 0.1*np.pi),
                'z': (-0.1*np.pi, 0.1*np.pi),
            },
            'weight': 1.0
        },
        'LHeel': {
            'limits': {
                'x': (-0.2*np.pi, 0.2*np.pi),
                'y': (-0.1*np.pi, 0.1*np.pi),
                'z': (-0.1*np.pi, 0.1*np.pi),
            },
            'weight': 1.0
        },
        'RHeel': {
            'limits': {
                'x': (-0.2*np.pi, 0.2*np.pi),
                'y': (-0.1*np.pi, 0.1*np.pi),
                'z': (-0.1*np.pi, 0.1*np.pi),
            },
            'weight': 1.0
        },
    }

def anatomical_pose_prior(body_pose):
    """
    Compute weighted violation of anatomical joint limits
    """
    batch_size = body_pose.shape[0]
    pose_mat = axis_angle_to_matrix(body_pose.reshape(-1, 3))
    euler_angles = matrix_to_euler_angles(pose_mat, "XYZ")
    euler_angles = euler_angles.reshape(batch_size, -1, 3)

    # Create reverse mapping from SMPL index to BODY25 joint name
    smpl_to_body25_joint = {}
    for body25_idx, joint_name in enumerate(joints_name):
        smpl_idx = int(joint_name.split("=>")[1].strip().split()[0]) if "=>" in joint_name else None
        if smpl_idx is not None and smpl_idx < 23:  # Only include indices within SMPL range
            smpl_to_body25_joint[smpl_idx] = joint_name

    joint_data = get_joint_limits_and_weights()
    violations = []
    
    for smpl_idx, joint_name in smpl_to_body25_joint.items():
        if joint_name not in joint_data:
            continue
            
        joint_angles = euler_angles[:, smpl_idx]
        data = joint_data[joint_name]
        
        for axis_idx, (min_angle, max_angle) in enumerate(data['limits'].values()):
            angle = joint_angles[..., axis_idx]
            
            # Compute violations
            min_violation = torch.clamp(min_angle - angle, min=0)
            max_violation = torch.clamp(angle - max_angle, min=0)
            violation = (min_violation + max_violation) * data['weight']
            
            violations.append(violation)
    
    if not violations:  # Handle case where no valid joints were processed
        return torch.tensor(0.0, device=body_pose.device, requires_grad=True)
        
    return torch.stack(violations).mean()

def angle_magnitude_prior(body_pose):
    """
    Penalize extremely large rotation magnitudes
    """
    # Angle-axis representation naturally encodes rotation magnitude
    rotation_magnitudes = torch.norm(body_pose.reshape(-1, 3), dim=1)
    threshold = torch.pi  # Maximum reasonable rotation
    excess = torch.clamp(rotation_magnitudes - threshold, min=0)
    return excess.mean()

def shape_parameter_regularization(betas, original_dim=10):
    """
    Apply stronger regularization to higher-order shape parameters
    """
    # Ensure betas has correct shape (add batch dim if needed)
    if len(betas.shape) == 1:
        betas = betas.unsqueeze(0)  # Add batch dimension if missing

    # Stronger regularization for additional parameters
    weights = [0.5, 1.5]
    primary_reg = betas[:, :original_dim].pow(2).mean()
    secondary_reg = betas[:, original_dim:].pow(2).mean()
    
    return (weights[0] * primary_reg + weights[1] * secondary_reg)/sum(weights)

def shape_specific_loss(silhouette_pred, silhouette_target, weight_masks=None):
    """Calculate shape-specific metrics with proper normalization"""
    # Normalize inputs to 0-1 range
    silhouette_pred = torch.clamp(silhouette_pred, 0, 1)
    silhouette_target = torch.clamp(silhouette_target, 0, 1)
    
    # Get image dimensions for normalization
    B, H, W = silhouette_pred.shape
    total_pixels = H * W
    
    if weight_masks is not None:
        # Convert weight_masks to tensor if it's numpy array
        if isinstance(weight_masks, np.ndarray):
            weight_masks = torch.from_numpy(weight_masks).float().to(DEVICE)
        
        # Apply weights to silhouettes
        weighted_pred = silhouette_pred * weight_masks
        weighted_target = silhouette_target * weight_masks
        
        # Calculate effective total pixels (accounting for weights)
        effective_pixels = weight_masks.sum(dim=[1, 2])
    else:
        weighted_pred = silhouette_pred
        weighted_target = silhouette_target
        effective_pixels = torch.full((B,), total_pixels, device=DEVICE)

    # 1. Normalized area difference (using weighted pixels)
    pred_area = weighted_pred.sum(dim=[1, 2]) / effective_pixels
    target_area = weighted_target.sum(dim=[1, 2]) / effective_pixels
    area_loss = F.mse_loss(pred_area, target_area)

    # 2. Normalized height/width proportions (using weighted pixels)
    pred_height = (weighted_pred.sum(dim=2) > 0).float().sum(dim=1) / H
    target_height = (weighted_target.sum(dim=2) > 0).float().sum(dim=1) / H
    height_loss = F.mse_loss(pred_height, target_height)

    pred_width = (weighted_pred.sum(dim=1) > 0).float().sum(dim=1) / W
    target_width = (weighted_target.sum(dim=1) > 0).float().sum(dim=1) / W
    width_loss = F.mse_loss(pred_width, target_width)

    # 3. Aspect ratio (using weighted dimensions)
    pred_aspect = pred_height / (pred_width + 1e-8)
    target_aspect = target_height / (target_width + 1e-8)
    aspect_loss = F.mse_loss(pred_aspect, target_aspect)

    total_shape_loss = (0.4 * area_loss + 
                       0.3 * height_loss + 
                       0.3 * width_loss + 
                       0.2 * aspect_loss)
    
    return total_shape_loss

def generate_and_save_hair_hat_face_weight_masks(
        input_image_dir: str, 
        input_image_names: List[str],
        output_weight_masks_dir: str, 
        model_type: ModelType, 
        downscale: int = 1,
        batch_size: int = BATCHSIZE 
    ) -> List[np.ndarray]:
    """Generate weight masks for hair, hat, and face regions with batch processing"""
    
    img_paths = [os.path.join(input_image_dir, name) for name in input_image_names]
    if not all(os.path.exists(img_path) for img_path in img_paths):
        raise FileNotFoundError(f"One or more images do not exist in {input_image_dir}")
    
    total_frames = len(img_paths)
    weights = [0.0, 0.5, 1.0]
    weight_masks = []
    
    print(f"\nProcessing {total_frames} frames in batches of {batch_size}")
    
    # Process in batches
    for i in range(0, total_frames, batch_size):
        batch_end = min(i + batch_size, total_frames)
        batch_paths = img_paths[i:batch_end]
        
        print(f"\nProcessing batch {i//batch_size + 1}/{(total_frames + batch_size - 1)//batch_size}")
        print(f"1. Running SCHP segmentation for frames {i} to {batch_end-1}")
        
        # Get segmentation maps for batch
        seg_maps = get_schp_segmentation_batch(model_type, batch_paths, batch_size)
        
        print(f"2. Generating weight masks for frames {i} to {batch_end-1}")
        for img_path, seg_map in tqdm(zip(batch_paths, seg_maps), 
                                    total=len(batch_paths),
                                    desc="Generating weight masks"):
            weight_mask = get_hair_hat_face_weight_mask(seg_map=seg_map, 
                                                      model_type=model_type, 
                                                      weights=weights)
            
            # Validate unique values
            print(f"Unique values in weight mask: {np.unique(weight_mask).tolist()}")
            assert np.unique(weight_mask).tolist() == weights, "Weight mask contains invalid values"

            # Save as data (value 0, 0.5, 1)
            basename = os.path.basename(img_path)
            weight_mask_data_name = os.path.splitext(basename)[0] + '.npy'
            np.save(os.path.join(output_weight_masks_dir, weight_mask_data_name), weight_mask)
            
            # Save visualization
            weight_mask_uint8 = np.round(weight_mask * 255).astype(np.uint8)
            weight_mask_png_name = os.path.splitext(basename)[0] + '.png'
            cv2.imwrite(os.path.join(output_weight_masks_dir, weight_mask_png_name), 
                        weight_mask_uint8)

            if downscale > 1:
                weight_mask = cv2.resize(weight_mask, dsize=None, fx=1/downscale, fy=1/downscale)
            
            weight_masks.append(weight_mask)
    
    return weight_masks

def main(root, keypoints_threshold, use_silhouette=True, gender="female", downscale=1, use_temporal_smoothness=False):
    camera = dict(np.load(f"{root}/cameras.npz"))
    if downscale > 1:
        camera["intrinsic"][:2] /= downscale
    projection_matrices = camera["intrinsic"] @ camera["extrinsic"][:3]
    projection_matrices = torch.from_numpy(projection_matrices).float().to(DEVICE)

    # Get image dimensions from camera intrinsics
    H, W = int(camera["intrinsic"][1, 2] * 2), int(camera["intrinsic"][0, 2] * 2)
    print(f"Image dimensions: {W}x{H}")

    # prepare data
    print("\nInitializing joint mapper and loading data...")
    joint_mapper = BODY25JointMapper()
    print(f"Joint mapper initialized: {joint_mapper is not None}")
    
    print("\nLoading SMPL params...")
    smpl_params = dict(np.load(f"{root}/poses.npz"))
    print(f"SMPL params loaded: {len(smpl_params) > 0}")
    print(f"SMPL params keys: {list(smpl_params.keys())}")
    
    print("\nLoading keypoints...")
    keypoints_2d = np.load(f"{root}/keypoints.npy")
    print(f"Keypoints loaded: {keypoints_2d is not None}")
    keypoints_2d = torch.from_numpy(keypoints_2d).float().to(DEVICE)
    print(f"Keypoints converted to tensor on {DEVICE}")

    # After loading data
    print("\nInput Data:")
    print(f"Number of frames: {len(keypoints_2d)}")
    print(f"Keypoints shape: {keypoints_2d.shape}")
    print(f"Initial SMPL params:")
    for k, v in smpl_params.items():
        print(f"  {k}: {v.shape} (min: {v.min():.3f}, max: {v.max():.3f})")

     # Modify the SMPL model initialization to use all shape parameters
    body_model = smplx.SMPL(
        "./data/SMPLX/smpl", 
        gender=gender,
        num_betas=300,
        use_hands=False,
        use_feet=False,
        use_face=False,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_transl=True,
    )
    body_model.to(DEVICE)
    body_model.train()  # Enable gradient computation

    # Verify model parameters
    print("\nSMPL Model Parameters:")
    for name, param in body_model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    # When loading or initializing betas, expand to 300 dimensions
    params = {}
    for k, v in smpl_params.items():
        if k == "thetas":
            tensor = torch.from_numpy(v[:, :3]).clone().to(DEVICE)
            params["global_orient"] = nn.Parameter(tensor, requires_grad=True)
            tensor = torch.from_numpy(v[:, 3:]).clone().to(DEVICE)
            params["body_pose"] = nn.Parameter(tensor, requires_grad=True)
        elif k == "betas":
            # Debug original betas
            print(f"\nDebug Shape Parameters:")
            print(f"Original betas shape: {v.shape}")
            
            # Move to device immediately after creation
            original_betas = torch.from_numpy(v).clone().to(DEVICE)
            if len(original_betas.shape) == 1:
                original_betas = original_betas.unsqueeze(0) # Add batch dimension if missing
            print(f"After unsqueeze shape: {original_betas.shape}")
            
            expanded_betas = torch.zeros((original_betas.shape[0], 300), device=DEVICE)
            expanded_betas[:, :original_betas.shape[1]] = original_betas  # Copy existing betas
            print(f"Expanded betas shape: {expanded_betas.shape}")

            # Initialize remaining betas with small random values
            expanded_betas[:, original_betas.shape[1]:] = torch.randn(
                (original_betas.shape[0], 300 - original_betas.shape[1]), 
                device=DEVICE) * 0.0001  # Small initialization
            
            # Check SMPL model's shape capabilities
            print("\nSMPL Model Shape Information:")
            print(f"Model's shape space dim: {body_model.SHAPE_SPACE_DIM}")
            print(f"Model's shapedirs shape: {body_model.shapedirs.shape}")
             # Verify the actual shape components available
            num_available_shapes = body_model.shapedirs.shape[-1]
            print(f"Number of available shape components: {num_available_shapes}")
            
            if num_available_shapes < 300:
                print("WARNING: Model does not support 300 PCs!")
                print("Using limited number of shape components.")
            else:
                print("Successfully using 300 PCs model!")
                
            params[k] = nn.Parameter(expanded_betas, requires_grad=True)
        else:
            tensor = torch.from_numpy(v).clone().to(DEVICE)
            params[k] = nn.Parameter(tensor, requires_grad=True)

    # Verify all parameters are properly set up
    for k, v in params.items():
        assert isinstance(v, nn.Parameter), f"{k} is not a Parameter"
        assert v.requires_grad, f"{k} does not require grad"
        assert v.device.type == DEVICE, f"{k} is not on {DEVICE}"

    # After creating all parameters
    for name, param in params.items():
        print(f"{name}:")
        print(f"  requires_grad: {param.requires_grad}")
        print(f"  device: {param.device}")
        print(f"  shape: {param.shape}")
        
    # Different learning rates for shape and pose
    optimizer_shape = torch.optim.Adam(
        [params['betas']], 
        lr=SHAPE_LR,  # Higher learning rate for shape
        betas=(0.9, 0.999) 
    )
    optimizer_pose = torch.optim.Adam(
        [params['body_pose'], params['global_orient'], params['transl']], 
        lr=POSE_LR,
        betas=(0.9, 0.999)
    )
    
    def closure(shape_only=False):
        smpl_output = body_model(**params)
        keypoints_pred = project(projection_matrices, joint_mapper(smpl_output))
        
        # Debug prints
        print(f"\nSMPL output requires grad: {smpl_output.vertices.requires_grad}")
        print(f"Keypoints pred requires grad: {keypoints_pred.requires_grad}")
        print("\nOptimization step:")
        print(f"SMPL output joints shape: {smpl_output.joints.shape}")
        print(f"Projected keypoints shape: {keypoints_pred.shape}")
        print(f"Target keypoints shape: {keypoints_2d[..., :2].shape}")

        # Add lower weights for head keypoints
        keypoint_weights = torch.ones_like(keypoints_2d[..., 0])
        keypoint_weights[..., HEAD_KEYPOINT_INDICES] = 0.01  # Reduce influence of head keypoints
        
        # Calculate raw error before normalization
        raw_diff = keypoints_2d[..., :2] - keypoints_pred
        raw_error = raw_diff.abs().sum(-1)
        print(f"Raw error stats - min: {raw_error.min():.6f}, max: {raw_error.max():.6f}, mean: {raw_error.mean():.6f}")
        
        # 1. Input normalization
        keypoints_norm = keypoints_2d[..., :2] / torch.tensor([W, H], device=DEVICE)
        keypoints_pred_norm = keypoints_pred / torch.tensor([W, H], device=DEVICE)
        
        # 2. Calculate difference and its magnitude
        diff = keypoints_norm - keypoints_pred_norm
        diff_norm = diff.abs().sum(-1)  # L1 norm of difference
        print(f"Normalized error stats - min: {diff_norm.min():.6f}, max: {diff_norm.max():.6f}, mean: {diff_norm.mean():.6f}")
        
        # Huber loss with correct dimensions
        error = torch.where(
            diff_norm < 1.0,
            0.5 * diff_norm.square(),
            diff_norm - 0.5
        )
        
        print(f"Error stats - min: {error.min():.6f}, max: {error.max():.6f}, mean: {error.mean():.6f}")
        
        # Apply confidence mask
        mask = (keypoints_2d[..., 2] > keypoints_threshold)
        print(f"Valid keypoints: {mask.sum().item()}/{mask.numel()}")
        error = error * mask.float() * keypoint_weights
        
        # 3. Log-space loss for better numerical stability
        keypoint_loss = torch.log1p(error[:, SELECT_JOINTS]).mean()
        shape_reg = shape_parameter_regularization(params['betas'][0])
        pose_prior = pose_prior_loss(params['body_pose'])

        # Only add temporal smoothness if enabled
        if use_temporal_smoothness:
            loss_downgrade = 0.1 # TODO: was 0.01 check whether it's needed?
            smoothness_loss = temporal_smoothness_loss(smpl_output.vertices, window_size=3, loss_downgrade=loss_downgrade)
            pose_smoothness = pose_smoothness_loss(params['body_pose'])

            # The weight of shape_reg 0.005 for shape regularization is a starting point - 
            # you may need to adjust this value based on your specific needs. 
            # A smaller value will allow more freedom in the shape parameters, 
            # while a larger value will constrain them more strongly.
            if shape_only:
                weights = [0.5, 0.001, 0.1, 0.0, 0.0] # Zero out pose-related weights
            else:
                weights = [0.5, 0.001, 0.1, 0.1, 0.1]
            total_loss = (weights[0] * keypoint_loss 
                         + weights[1] * shape_reg 
                         + weights[2] * smoothness_loss 
                         + weights[3] * pose_smoothness 
                         + weights[4] * pose_prior)
            print(f"Total loss: {total_loss.item():.6f}: \n - Keypoint loss: {keypoint_loss.item():.6f}, weighted keypoint loss: {weights[0] * keypoint_loss.item():.6f}, "
                  f"\n - Shape regularization: {shape_reg.item():.6f}, weighted shape regularization: {weights[1] * shape_reg.item():.6f}, "
                  f"\n - Smoothness loss: {smoothness_loss.item():.6f}, weighted smoothness loss: {weights[2] * smoothness_loss.item():.6f}, "
                  f"\n - Pose smoothness: {pose_smoothness.item():.6f}, weighted pose smoothness: {weights[3] * pose_smoothness.item():.6f}, "
                  f"\n - Pose prior: {pose_prior.item():.6f}, weighted pose prior: {weights[4] * pose_prior.item():.6f}")
            
        else:
            # The weight of shape_reg 0.005 for shape regularization is a starting point - 
            # you may need to adjust this value based on your specific needs. 
            # A smaller value will allow more freedom in the shape parameters, 
            # while a larger value will constrain them more strongly.
            if shape_only:
                weights = [0.5, 0.001, 0.0] # Zero out pose-related weights
            else:
                weights = [0.5, 0.001, 0.1]
            total_loss = (weights[0] * keypoint_loss 
                         + weights[1] * shape_reg 
                         + weights[2] * pose_prior)
            print(f"Total loss: {total_loss.item():.6f}: \n - Keypoint loss: {keypoint_loss.item():.6f}, weighted keypoint loss: {weights[0] * keypoint_loss.item():.6f}, "
                  f"\n - Shape regularization: {shape_reg.item():.6f}, weighted shape regularization: {weights[1] * shape_reg.item():.6f}, "
                  f"\n - Pose prior: {pose_prior.item():.6f}, weighted pose prior: {weights[2] * pose_prior.item():.6f}")

        if torch.isnan(total_loss):
            print("WARNING: NaN loss detected!")
            raise ValueError("NaN loss detected!")
        return total_loss
    
    optimize(optimizer_shape=optimizer_shape, 
             optimizer_pose=optimizer_pose, 
             closure=closure, 
             params=params, 
             batch_start=0, 
             batch_end=len(keypoints_2d), 
             max_iter=100)

    def create_headless_mesh(smpl_output, body_model):
        vertices = smpl_output.vertices.clone()
        batch_size = vertices.shape[0]
        
        # Get neck joint position (joint 12 in SMPL)
        neck_joint_y = smpl_output.joints[0, 12, 1]
        
        # Create mask for vertices above neck joint
        vertices_above_neck = vertices[0, :, 1] < neck_joint_y
        HEAD_VERTEX_IDS = torch.where(vertices_above_neck)[0]
        
        # Create faces mask (faces containing head vertices)
        faces = body_model.faces_tensor[None].repeat(batch_size, 1, 1)
        head_faces_mask = torch.any(torch.isin(faces, HEAD_VERTEX_IDS), dim=-1)
        
        # Remove head faces by creating new faces tensor without head faces
        valid_faces = ~head_faces_mask[0]  # Take first batch element since mask is same for all
        faces_no_head = faces[0, valid_faces, :][None].repeat(batch_size, 1, 1)
        
        # Only keep vertices that are used by remaining faces
        used_vertices = torch.unique(faces_no_head[0])
        vertices_no_head = vertices[:, used_vertices, :]
        
        # Update face indices to match new vertex ordering
        vertex_map = torch.zeros(vertices.shape[1], dtype=torch.long, device=DEVICE)
        vertex_map[used_vertices] = torch.arange(len(used_vertices), device=DEVICE)
        faces_no_head = vertex_map[faces_no_head]
        
        return vertices_no_head, faces_no_head

    if use_silhouette:
        SIL_DEBUG = True
        # Load pre-generated body-only masks
        if NO_HEAD_SILHOUETTE:
            mask_paths = sorted(glob.glob(f"{root}/body_only_masks_schp/*")) # headless masks
        else:
            mask_paths = sorted(glob.glob(f"{root}/masks/*")) # full body masks

        masks = [cv2.imread(p)[..., 0] for p in mask_paths]
        if downscale > 1:
            masks = [cv2.resize(m, dsize=None, fx=1/downscale, fy=1/downscale) 
                    for m in masks]
        masks = np.stack(masks, axis=0)
        print(f"\nSilhouette Optimization:")
        print(f"Number of masks: {len(masks)}")
        print(f"Mask shape: {masks[0].shape}")

        if USE_HAIR_HAT_FACE_WEIGHT:
            masked_image_paths = sorted(glob.glob(f"{root}/masked_images/*"))
            masked_image_names = [os.path.basename(p) for p in masked_image_paths]
            masked_image_base_names = [os.path.splitext(n)[0] for n in masked_image_names]

            hhf_weight_masks_dir = f"{root}/hhf_weight_masks"
            os.makedirs(hhf_weight_masks_dir, exist_ok=True)
            hhf_weight_mask_names = [n + '.npy' for n in masked_image_base_names]

            existing_weight_mask_paths = sorted(glob.glob(f"{hhf_weight_masks_dir}/*.npy"))

            new_weight_mask_indices = [i for i, n in enumerate(hhf_weight_mask_names) if n not in existing_weight_mask_paths]

            input_image_names = [masked_image_names[i] for i in new_weight_mask_indices]

            # load existing weight masks
            for i, p in enumerate(existing_weight_mask_paths):
                if os.path.exists(p):
                    print(f"Loading existing hair, hat, and face weight mask from {p}")
                    weight_masks[i] = np.load(p)
                    if downscale > 1:
                        weight_masks[i] = cv2.resize(weight_masks[i], dsize=None, fx=1/downscale, fy=1/downscale)

            # generate new weight masks
            print(f"\nGenerating hair, hat, and face weight masks and saving to {hhf_weight_masks_dir}")
            model_type = ModelType.ATR
            new_weight_masks = generate_and_save_hair_hat_face_weight_masks(input_image_dir=f"{root}/masked_images",
                                                                        input_image_names=input_image_names,
                                                                        output_weight_masks_dir=hhf_weight_masks_dir, 
                                                                        model_type=model_type, 
                                                                        downscale=downscale,
                                                                        batch_size=BATCHSIZE)
            weight_masks[new_weight_mask_indices] = new_weight_masks

            weight_masks = np.stack(weight_masks, axis=0)

            # Convert to tensor and move to GPU
            weight_masks = torch.from_numpy(weight_masks).float().to(DEVICE)
            print(f"Weight masks shape: {weight_masks.shape}")
            print(f"Number of weight masks: {len(weight_masks)}")

        # Generate smpl projected silhouette
        img_size = masks[0].shape[:2]
        renderer = build_renderer(camera, img_size)

        if SIL_DEBUG:
            # Create debug directory
            debug_dir = os.path.join(root, "refine-smpl", "debug")
            os.makedirs(debug_dir, exist_ok=True)

        # Process frames in small batches to improve speed while maintaining memory efficiency
        for i in range(0, len(masks), BATCHSIZE):
            batch_end = min(i + BATCHSIZE, len(masks))
            batch_masks = torch.from_numpy(masks[i:batch_end]).float().to(DEVICE) / 255
            
            optimizer_shape = torch.optim.Adam(
                [params['betas']], 
                lr=SHAPE_LR,
                betas=(0.9, 0.999)
            )
            optimizer_pose = torch.optim.Adam(
                [params['body_pose'], params['global_orient'], params['transl']], 
                lr=POSE_LR,
                betas=(0.9, 0.999)
            )
            
            def closure(shape_only=False):
                # Get SMPL output for batch
                with torch.set_grad_enabled(True):  # Explicitly enable gradients
                    smpl_output = body_model(
                        betas=params["betas"],  # Remove clone().detach() to maintain gradient flow
                        global_orient=params["global_orient"][i:batch_end],
                        body_pose=params["body_pose"][i:batch_end],
                        transl=params["transl"][i:batch_end],
                        return_verts=True,
                        return_full_pose=True,
                    )
                if not smpl_output.vertices.requires_grad:
                    raise ValueError("SMPL vertices lost gradient tracking!")
                

                if NO_HEAD_SILHOUETTE:
                    # headless version
                    vertices_no_head, faces_no_head = create_headless_mesh(smpl_output, body_model)
                    vertices = vertices_no_head.float()
                    faces = faces_no_head
                else: # full body
                    vertices = smpl_output.vertices.float()
                    faces = body_model.faces_tensor
                
                # Add batch dimension to faces if it's missing
                if len(faces.shape) == 2:
                    faces = faces.unsqueeze(0).expand(vertices.shape[0], -1, -1)

                # Verify shapes before creating meshes
                print(f"Vertices shape: {vertices.shape}")  # Should be (batch_size, num_vertices, 3)
                print(f"Faces shape: {faces.shape}")       # Should be (batch_size, num_faces, 3)

                # Create meshes
                meshes = Meshes(verts=vertices, faces=faces).to(DEVICE)
                # Render silhouettes
                silhouette = renderer(meshes)[..., 3]

                # Ensure proper normalization
                silhouette = torch.clamp(silhouette, 0, 1)
                # Create and normalize local copy of batch_masks
                batch_masks_local = batch_masks.clone()
                batch_masks_local = torch.clamp(batch_masks_local, 0, 1)

                # Compare silhouette with the mask
                if USE_HAIR_HAT_FACE_WEIGHT:
                    # Get the correct batch of weight masks
                    batch_weight_masks = weight_masks[i:batch_end]
                    # Ensure shapes match
                    if batch_weight_masks.shape[0] != silhouette.shape[0]:
                        print(f"Warning: batch size mismatch. Weight masks: {batch_weight_masks.shape}, Silhouette: {silhouette.shape}")
                        # If needed, repeat the weight masks to match batch size
                        if batch_weight_masks.shape[0] == 1:
                            batch_weight_masks = batch_weight_masks.repeat(silhouette.shape[0], 1, 1)
                    
                    loss_silhouette = F.mse_loss(
                        batch_masks_local * batch_weight_masks, 
                        silhouette * batch_weight_masks
                    )
                else:
                    loss_silhouette = F.mse_loss(batch_masks_local, silhouette)

                # Loss computations
                shape_reg = shape_parameter_regularization(params['betas'][0])
                shape_loss = shape_specific_loss(silhouette, batch_masks_local, weight_masks)
                
                # Debug visualization with reduced frequency
                if SIL_DEBUG and i % 20 == 0:  # Every 20th frame
                    save_sil_debug_image(silhouette, batch_masks_local)

                # Rest of loss computation...
                keypoints_pred = project(projection_matrices, joint_mapper(smpl_output))
                loss_keypoints = (keypoints_2d[i:batch_end, :, :2] - keypoints_pred).square().sum(-1).sqrt()
                
                m1 = (keypoints_2d[i:batch_end, :, 2] > 0)
                loss_keypoints = (loss_keypoints * m1.float()).mean()

                anatomical_loss = anatomical_pose_prior(params['body_pose'][i:batch_end])
                magnitude_prior = angle_magnitude_prior(params['body_pose'])
                
                # Final loss computation
                if shape_only:
                    weights = [50.0, 0, 0.001, 0.0, 0.0, 0.0] # [50.0, 500, 0.001, 0.0, 0.0, 0.0] # TODO: removed shape_loss temporarily. fix later.
                else:
                    weights = [50.0, 0, 0.001, 0.01, 0.5, 0.5] # [50.0, 500, 0.001, 0.01, 0.5, 0.5] # TODO: removed shape_loss temporarily. fix later.
                
                loss = (weights[0] * loss_silhouette 
                        + weights[1] * shape_loss
                        + weights[2] * shape_reg
                        + weights[3] * loss_keypoints 
                        + weights[4] * anatomical_loss 
                        + weights[5] * magnitude_prior)

                # Verify final loss requires gradient
                print("\nFinal loss status:")
                print(f"Final loss requires_grad: {loss.requires_grad}")
                
                if not loss.requires_grad:
                    raise ValueError("Final loss doesn't require gradients! Check computation graph.")

                print(f"Loss: {loss.item():.6f}: \n - Silhouette: {loss_silhouette.item():.6f}, weighted loss_silhouette: {weights[0] * loss_silhouette.item():.6f}, "
                      f"\n - Shape specific: {shape_loss.item():.6f}, weighted shape_loss: {weights[1] * shape_loss.item():.6f}, "
                      f"\n - Shape regularization: {shape_reg.item():.6f}, weighted shape regularization: {weights[2] * shape_reg.item():.6f}, "
                      f"\n - Keypoints: {loss_keypoints.item():.6f}, weighted loss_keypoints: {weights[3] * loss_keypoints.item():.6f}, "
                      f"\n - Anatomical: {anatomical_loss.item():.6f}, weighted loss_anatomical: {weights[4] * anatomical_loss.item():.6f}, "
                      f"\n - Magnitude: {magnitude_prior.item():.6f}, weighted loss_magnitude: {weights[5] * magnitude_prior.item():.6f}")
                return loss
            
            @torch.no_grad()
            def save_sil_debug_image(silhouette, batch_masks_local):
                batch_idx = 0  # Use first frame in current batch
                # Convert silhouette to numpy and scale to 0-255 range
                sil = silhouette[batch_idx].detach().cpu().numpy()
                sil = (sil * 255).astype(np.uint8)
                        
                # Get the corresponding mask
                mask = batch_masks_local[batch_idx].detach().cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                
                # Create masked image (red = silhouette, green = mask)
                masked_img = np.zeros((sil.shape[0], sil.shape[1], 3), dtype=np.uint8)
                masked_img[..., 2] = sil  # Red channel for silhouette
                masked_img[..., 1] = mask         # Green channel for mask
                
                # Save the silhouette
                debug_path_sil = os.path.join(debug_dir, f'silhouette_{i:04d}.png')
                debug_path_masked = os.path.join(debug_dir, f'masked_{i:04d}.png')

                cv2.imwrite(debug_path_sil, sil)
                cv2.imwrite(debug_path_masked, masked_img)
                print(f"Saved silhouette for frame {i}")

            optimize(optimizer_shape=optimizer_shape, 
                     optimizer_pose=optimizer_pose, 
                     closure=closure, 
                     params=params, 
                     batch_start=i, 
                     batch_end=batch_end, 
                     max_iter=150)

    smpl_params = dict(smpl_params)
    for k in smpl_params:
        if k == "betas":
            smpl_params[k] = params[k][0].detach().cpu().numpy()
        elif k == "thetas":
            smpl_params[k][:, :3] = params["global_orient"].detach().cpu().numpy()
            smpl_params[k][:, 3:] = params["body_pose"].detach().cpu().numpy()
        elif k == "body_pose":
            smpl_params[k] = params[k].detach().cpu().numpy()
            smpl_params[k][:, -12:] = 0
        else:
            smpl_params[k] = params[k].detach().cpu().numpy()
    
    # Replace the existing measurement code with:
    save_mesh_and_measurements(body_model, params, root)
    
    np.savez(f"{root}/poses_optimized.npz", **smpl_params)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--gender", type=str, default="female")
    parser.add_argument("--keypoints-threshold", type=float, default=0.2)
    parser.add_argument("--silhouette", action="store_true")
    parser.add_argument("--downscale", type=float, default=1.0)
    parser.add_argument('--use_temporal_smoothness', action='store_true', 
                      help='Enable temporal smoothness loss')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir, 
         args.keypoints_threshold, 
         args.silhouette, 
         args.gender, 
         args.downscale,
         args.use_temporal_smoothness)
