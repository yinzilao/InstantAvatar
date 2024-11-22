import numpy as np
import smplx
import torch
import trimesh
import os
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_smpl_params(npz_path):
    """Load SMPL parameters from npz file"""
    params = dict(np.load(npz_path))
    
    # Convert numpy arrays to torch tensors
    smpl_params = {}
    for k, v in params.items():
        tensor = torch.from_numpy(v).float().to(DEVICE)
        if k == "thetas":
            smpl_params["global_orient"] = tensor[:, :3]
            smpl_params["body_pose"] = tensor[:, 3:]
        else:
            smpl_params[k] = tensor
            
    return smpl_params

def save_posed_mesh(body_model, params, output_path, pose_type='zero'):
    """Save SMPL mesh in A-pose, T-pose, or zero pose
    
    Args:
        body_model: SMPL model instance
        params: Dict containing optimized SMPL parameters
        output_path: Path to save the mesh
        pose_type: default is zero pose, 'a' for A-pose, 't' for T-pose
    """
    # Keep optimized shape parameters
    new_params = {
        'betas': params['betas'],
        'global_orient': torch.zeros_like(params['global_orient']),
        'transl': torch.zeros_like(params['transl'])
    }

    body_pose = torch.zeros_like(params['body_pose'][0])
    
    if pose_type.lower() == 't':
        # T-pose: arms straight out (90 degrees)
        body_pose[16*3 + 1] = np.pi/2  # Left shoulder Y-axis
        body_pose[17*3 + 1] = -np.pi/2  # Right shoulder Y-axis
        
    elif pose_type.lower() == 'a':
        # A-pose: arms ~45 degrees down
        body_pose[16*3 + 1] = np.pi/4  # Left shoulder Y-axis
        body_pose[17*3 + 1] = -np.pi/4  # Right shoulder Y-axis
    else:
        print(f"WARNING: Unknown pose type: {pose_type}, falling back to default 'zero' pose.")

    new_params['body_pose'] = body_pose.unsqueeze(0)
    
    # Generate mesh with new pose
    smpl_output = body_model(**new_params)
    vertices = smpl_output.vertices.detach().cpu().numpy()
    faces = body_model.faces

    # Save mesh
    mesh = trimesh.Trimesh(vertices[0], faces)
    mesh.export(output_path)
    print(f"Saved {pose_type}-pose mesh to: {output_path}")

def main(args):
    # Initialize SMPL model
    body_model = smplx.SMPL(
        args.model_path,
        gender=args.gender,
        num_betas=300,
    ).to(DEVICE)

    # Load optimized parameters
    params = load_smpl_params(args.input_npz)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save meshes in different poses
    poses = ['zero', 'a', 't']
    for pose_type in poses:
        output_path = os.path.join(args.output_dir, f'body_mesh_{pose_type}_pose.obj')
        save_posed_mesh(body_model, params, output_path, pose_type)

def parse_args():
    parser = argparse.ArgumentParser(description='Save SMPL mesh in different poses')
    parser.add_argument('--input_npz', type=str, required=True,
                        help='Path to poses_optimized.npz file')
    parser.add_argument('--model_path', type=str, default='./data/SMPLX/smpl',
                        help='Path to SMPL model folder')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output meshes')
    parser.add_argument('--gender', type=str, default='neutral',
                        choices=['neutral', 'male', 'female'],
                        help='SMPL model gender')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args) 