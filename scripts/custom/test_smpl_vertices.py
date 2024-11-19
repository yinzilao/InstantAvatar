import numpy as np
import smplx
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    BlendParams,
    SoftSilhouetteShader,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_renderer(img_size=512):
    # Simple camera settings
    focal_length = 1000
    principal_point = (img_size//2, img_size//2)
    
    cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=(principal_point,),
        image_size=((img_size, img_size),),
        in_ndc=False,
        device=DEVICE
    )
    
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=100,
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

def main():
    # Initialize SMPL model
    body_model = smplx.SMPL("./data/SMPLX/smpl", gender="neutral").to(DEVICE)
    
    # Create default pose
    batch_size = 1
    body_pose = torch.zeros((batch_size, 69), device=DEVICE)
    global_orient = torch.tensor([[0, np.pi, 0]], device=DEVICE)  # Remove extra dimension
    transl = torch.tensor([[0, 0, 2.5]], device=DEVICE)  # Move model back for better view
    
    # Get SMPL output with joints
    smpl_output = body_model(
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        return_joints=True  # Make sure we get joints
    )
    
    vertices = smpl_output.vertices[0].detach().cpu().numpy()
    joints = smpl_output.joints[0].detach().cpu().numpy()
    
    # Get neck joint position (joint 12 in SMPL)
    neck_joint_y = joints[12, 1]  # Y coordinate of neck joint
    
    # Silhouette without head
    vertices_no_head = smpl_output.vertices.clone()
    
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
    
    # Create meshes with modified vertices and faces
    meshes_no_head = Meshes(
        verts=vertices_no_head,
        faces=faces_no_head,
    ).to(DEVICE)
    
    # For debugging, let's visualize the neck joint position
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='o', s=1)
    
    # Plot neck joint
    ax.scatter(joints[12, 0], joints[12, 1], joints[12, 2], c='g', marker='o', s=100, label='Neck Joint')
    
    # Plot removed vertices
    removed_vertices = vertices[HEAD_VERTEX_IDS]
    ax.scatter(removed_vertices[:, 0], removed_vertices[:, 1], removed_vertices[:, 2], 
              c='r', marker='o', s=50, label='Removed Vertices')
    
    ax.legend()
    plt.savefig('neck_joint_debug.png')
    plt.close()
    
    # 2. Visualize silhouettes (with and without head)
    renderer = build_renderer()
    
    # Original silhouette
    meshes = Meshes(
        verts=smpl_output.vertices,
        faces=body_model.faces_tensor[None].repeat(1, 1, 1),
    ).to(DEVICE)
    
    silhouette_original = renderer(meshes)[..., 3]
    silhouette_original = silhouette_original.detach().cpu().numpy()[0]
    
    # Render the modified mesh
    silhouette_no_head = renderer(meshes_no_head)[..., 3]
    silhouette_no_head = silhouette_no_head.detach().cpu().numpy()[0]
    
    # Save silhouettes
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(silhouette_original)
    plt.title('Original Silhouette')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(silhouette_no_head)
    plt.title('Silhouette without Head')
    plt.axis('off')
    
    plt.savefig('silhouettes_comparison.png')
    plt.close()

if __name__ == "__main__":
    main() 