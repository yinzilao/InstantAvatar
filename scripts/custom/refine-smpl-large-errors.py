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

DEVICE = "cuda"


def optimize(optimizer, closure, max_iter=10):
    pbar = tqdm(range(max_iter))
    prev_loss = float('inf')
    best_loss = None
    patience = 3
    patience_counter = 0
    min_improvement = 1e-6
    
    for i in pbar:
        loss = optimizer.step(closure)
        current_loss = loss.detach().cpu().numpy()

        # Initialize best_loss on first iteration
        if best_loss is None:
            best_loss = current_loss
        
        # Early stopping
        improvement = (prev_loss - current_loss) / (prev_loss + 1e-10)  # Avoid division by zero
        if abs(improvement) < min_improvement:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at iteration {i}, best loss: {best_loss:.6f}")
                break
        else:
            patience_counter = 0
            if current_loss < best_loss:
                best_loss = current_loss
            
        prev_loss = current_loss
        pbar.set_postfix_str(f"loss: {current_loss:.6f}")

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

@torch.no_grad()
def main(root, keypoints_threshold, use_silhouette, gender="female", downscale=1, use_temporal_smoothness=False):
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

    params = {}
    for k, v in smpl_params.items():
        if k == "thetas":
            tensor = torch.from_numpy(v[:, :3]).clone().to(DEVICE)
            params["global_orient"] = nn.Parameter(tensor)
            tensor = torch.from_numpy(v[:, 3:]).clone().to(DEVICE)
            params["body_pose"] = nn.Parameter(tensor)
        elif k == "betas":
            tensor = torch.from_numpy(v).clone().to(DEVICE)
            params[k] = nn.Parameter(tensor[None])
            # params[k] = tensor[None]
        else:
            tensor = torch.from_numpy(v).clone().to(DEVICE)
            params[k] = nn.Parameter(tensor)

    body_model = smplx.SMPL("./data/SMPLX/smpl", gender=gender)
    body_model.to(DEVICE)

    # optimize with keypoints
    optimizer = torch.optim.Adam(params.values(), lr=1e-3)
    def closure():
        optimizer.zero_grad()
        smpl_output = body_model(**params)
        keypoints_pred = project(projection_matrices, joint_mapper(smpl_output))
        
        # Debug prints
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
        loss = torch.log1p(error[:, SELECT_JOINTS]).mean()
        pose_prior = pose_prior_loss(params['body_pose'])

        # Only add temporal smoothness if enabled
        if use_temporal_smoothness:
            loss_downgrade = 0.1 # TODO: was 0.01 check whether it's needed?
            smoothness_loss = temporal_smoothness_loss(smpl_output.vertices, window_size=3, loss_downgrade=loss_downgrade)
            pose_smoothness = pose_smoothness_loss(params['body_pose'])

            total_loss = 0.84 * loss + 0.05 * smoothness_loss + 0.01 * pose_smoothness + 0.1 * pose_prior
            print(f"Total loss: {total_loss.item():.6f}: Keypoint loss: {loss.item():.6f}, "
                  f"Smoothness loss: {smoothness_loss.item():.6f}, "
                  f"Pose smoothness: {pose_smoothness.item():.6f}, "
                  f"Pose prior: {pose_prior.item():.6f}")
            
        else:
            total_loss = 0.9 * loss + 0.1 * pose_prior
            print(f"Total loss: {total_loss.item():.6f}: Keypoint loss: {loss.item():.6f}, "
                  f"Pose prior: {pose_prior.item():.6f}")

        if torch.isnan(total_loss):
            print("WARNING: NaN loss detected!")
            raise ValueError("NaN loss detected!")
            return torch.tensor(0.0, requires_grad=True, device=DEVICE)
        
        total_loss.backward()
        
        # 5. Gradient clipping with a smaller norm
        torch.nn.utils.clip_grad_norm_(params.values(), max_norm=0.1)

        return total_loss
    
    optimize(optimizer, closure, max_iter=200)

    # After first optimization
    smpl_output = body_model(**params)
    print("\nAfter Keypoint Optimization:")
    print(f"Vertices shape: {smpl_output.vertices.shape}")
    print(f"Joints shape: {smpl_output.joints.shape}")
    for k, v in params.items():
        print(f"  {k}: {v.shape} (min: {v.min().item():.3f}, max: {v.max().item():.3f})")

    def create_headless_mesh(smpl_output, body_model):
        vertices_no_head = smpl_output.vertices.clone()
        
        # Get neck joint position (joint 12 in SMPL)
        neck_joint_y = smpl_output.joints[0, 12, 1]  # Y coordinate of neck joint
        
        # Create mask for vertices above neck joint
        vertices_above_neck = vertices_no_head[0, :, 1] > neck_joint_y
        HEAD_VERTEX_IDS = torch.where(vertices_above_neck)[0].tolist()
        
        # Move head vertices far away
        vertices_no_head[..., HEAD_VERTEX_IDS, :] = torch.tensor([0., -1000., -1000.], device=DEVICE)
        
        # Get faces that contain any of the head vertices
        faces = body_model.faces_tensor[None].repeat(1, 1, 1)
        head_faces_mask = torch.any(torch.isin(faces, torch.tensor(HEAD_VERTEX_IDS, device=DEVICE)), dim=-1)
        
        # Remove faces connected to head vertices
        faces_no_head = faces.clone()
        faces_no_head[..., head_faces_mask, :] = 0
        
        return vertices_no_head, faces_no_head

    if use_silhouette:
        silhouette_debug = True
        # Load pre-generated body-only masks
        masks = sorted(glob.glob(f"{root}/body_only_masks_schp/*"))
        masks = [cv2.imread(p)[..., 0] for p in masks]
        if downscale > 1:
            masks = [cv2.resize(m, dsize=None, fx=1/downscale, fy=1/downscale) 
                    for m in masks]
        
        masks = np.stack(masks, axis=0)

        print(f"\nSilhouette Optimization:")
        print(f"Number of masks: {len(masks)}")
        print(f"Mask shape: {masks[0].shape}")

        img_size = masks[0].shape[:2]
        renderer = build_renderer(camera, img_size)

        if silhouette_debug:
            # Create debug directory
            debug_dir = os.path.join(root, "refine-smpl", "debug")
            os.makedirs(debug_dir, exist_ok=True)

        # Process frames in small batches to improve speed while maintaining memory efficiency
        BATCH_SIZE = 4  # Conservative batch size, increase if memory allows
        for i in range(0, len(masks), BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, len(masks))
            batch_masks = torch.from_numpy(masks[i:batch_end]).float().to(DEVICE) / 255
            
            optimizer = torch.optim.Adam(params.values(), lr=1e-3)
            
            def closure():
                optimizer.zero_grad()

                # Get SMPL output for batch
                smpl_output = body_model(
                    betas=params["betas"].clone().detach(),
                    global_orient=params["global_orient"][i:batch_end],
                    body_pose=params["body_pose"][i:batch_end],
                    transl=params["transl"][i:batch_end],
                )

                # Create headless mesh
                vertices_no_head, faces_no_head = create_headless_mesh(smpl_output, body_model)

                # Create meshes for both full body and headless
                meshes_full = Meshes(
                    verts=smpl_output.vertices.float(),  # Ensure float32
                    faces=body_model.faces_tensor[None].repeat(batch_end - i, 1, 1),
                ).to(DEVICE)

                meshes_no_head = Meshes(
                    verts=vertices_no_head.float(),  # Ensure float32
                    faces=faces_no_head,
                ).to(DEVICE)

                silhouette_full = renderer(meshes_full)[..., 3]
                silhouette_no_head = renderer(meshes_no_head)[..., 3]

                # Fix: Ensure batch dimensions match
                if silhouette_full.shape[0] != batch_masks.shape[0]:
                    batch_masks = batch_masks.expand(silhouette_full.shape[0], -1, -1)

                # Compare both silhouettes with the mask
                loss_silhouette_full = F.mse_loss(batch_masks, silhouette_full)
                loss_silhouette_no_head = F.mse_loss(batch_masks, silhouette_no_head)

                # Debug visualization with reduced frequency
                if silhouette_debug and i % 20 == 0:  # Reduced from every 10th to every 20th frame
                    with torch.no_grad():
                        # Convert tensors to numpy for visualization
                        sil_full = silhouette_full.detach().cpu().numpy()[0]
                        sil_no_head = silhouette_no_head.detach().cpu().numpy()[0]
                        mask_np = batch_masks.detach().cpu().numpy()[0]
                        
                        # Create difference images
                        diff_full = np.abs(sil_full - mask_np)
                        diff_no_head = np.abs(sil_no_head - mask_np)
                        
                        # Create visualization grid
                        plt.figure(figsize=(20, 10))
                        
                        plt.subplot(231)
                        plt.imshow(mask_np)
                        plt.title('Target Mask')
                        plt.axis('off')
                        
                        plt.subplot(232)
                        plt.imshow(sil_full)
                        plt.title('Full Silhouette')
                        plt.axis('off')
                        
                        plt.subplot(233)
                        plt.imshow(sil_no_head)
                        plt.title('No-Head Silhouette')
                        plt.axis('off')
                        
                        plt.subplot(235)
                        plt.imshow(diff_full, cmap='hot')
                        plt.title('Full Difference')
                        plt.colorbar()
                        plt.axis('off')
                        
                        plt.subplot(236)
                        plt.imshow(diff_no_head, cmap='hot')
                        plt.title('No-Head Difference')
                        plt.colorbar()
                        plt.axis('off')
                        
                        plt.suptitle(f'Frame {i} - Loss Full: {loss_silhouette_full:.4f}, Loss No-Head: {loss_silhouette_no_head:.4f}')
                        plt.tight_layout()
                        plt.savefig(os.path.join(debug_dir, f'silhouette_comparison_{i:04d}.png'))
                        plt.close()

                # Use the better matching silhouette
                loss_silhouette = torch.minimum(loss_silhouette_full, loss_silhouette_no_head)

                # Rest of the loss computation
                keypoints_pred = project(projection_matrices, joint_mapper(smpl_output))
                loss_keypoints = (keypoints_2d[i:batch_end, :, :2] - keypoints_pred).square().sum(-1).sqrt()
                
                m1 = (keypoints_2d[i:batch_end, :, 2] > 0)
                loss_keypoints = (loss_keypoints * m1.float()).mean()

                anatomical_loss = anatomical_pose_prior(params['body_pose'][i:batch_end])
                magnitude_prior = angle_magnitude_prior(params['body_pose'])
                
                loss = (0.1 * loss_silhouette 
                        + 0.7 * loss_keypoints 
                        + 0.15 * anatomical_loss 
                        + 0.05 * magnitude_prior)
                loss.backward()
                return loss

            optimize(optimizer, closure, max_iter=10)

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
