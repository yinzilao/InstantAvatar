import glob
import os
import torch
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np
from instant_avatar.utils.marching_cubes import marching_cubes
from pytorch3d.transforms import axis_angle_to_matrix
import kaolin.ops.mesh
from pytorch3d import ops
import numpy as np
import trimesh

def get_smpl_bounds(model):
    """Get bounding box from SMPL model vertices"""
    with torch.no_grad():
        # First initialize the deformer if not already done
        if not model.deformer.initialized:
            smpl_params = {
                "betas": model.SMPL_param.betas.weight,  # [1, 10]
                "body_pose": model.SMPL_param.body_pose.weight[0:1],  # [1, 69]
                "global_orient": model.SMPL_param.global_orient.weight[0:1],  # [1, 3]
                "transl": model.SMPL_param.transl.weight[0:1]  # [1, 3]
            }
            model.deformer.prepare_deformer(smpl_params)
        
        # Get canonical vertices from vs_template
        vertices = model.deformer.vs_template  # Should be [1, N, 3]
        
        # Calculate bounds with a safety margin
        min_bounds = vertices.min(dim=1)[0] - 0.3  # Add 30cm margin
        max_bounds = vertices.max(dim=1)[0] + 0.3
        
        print(f"SMPL bounds: min {min_bounds}, max {max_bounds}")
        return torch.stack([min_bounds, max_bounds]).to("cuda")

def get_density_stats(model, bbox, query_func, sample_points=10000):
    """Sample density values to determine appropriate level set threshold"""
    with torch.no_grad():
        # Sample random points within bbox
        points = torch.rand(sample_points, 3, device="cuda")
        points = points * (bbox[1] - bbox[0]) + bbox[0]
        
        # Query density at these points
        density = query_func(points)
        
        # Calculate statistics
        mean = density.mean().item()
        std = density.std().item()
        min_val = density.min().item()
        max_val = density.max().item()
        
        print(f"Density stats: mean={mean:.3f}, std={std:.3f}")
        print(f"Density range: [{min_val:.3f}, {max_val:.3f}]")
        
        # Set threshold at mean + 0.5 std deviation
        level_set = mean + 0.5 * std
        print(f"Suggested level_set: {level_set:.3f}")
        
        return level_set

def interpolate_vertices(source_verts, target_verts):
    """Interpolate source vertices to match target vertices using nearest neighbors"""
    # Find nearest neighbors between source and target vertices
    dist, idx, _ = ops.knn_points(source_verts.unsqueeze(0), target_verts.unsqueeze(0), K=3)
    
    # Calculate weights based on inverse distance
    weights = 1.0 / (dist + 1e-8)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    
    # Gather nearest neighbor vertices
    nearest_verts = target_verts[idx.squeeze(0)]
    
    # Compute weighted average
    interpolated = (nearest_verts * weights.unsqueeze(-1)).sum(dim=1)
    
    return interpolated

def extract_canonical_mesh(model, level_set=0.0, base_resolution=256):
    """Extract mesh in canonical T-pose space using multi-scale approach"""
    with torch.no_grad():
        # 1. Initialize canonical space with T-pose
        t_pose_params = {
            "betas": model.SMPL_param.betas.weight,
            "body_pose": torch.zeros_like(model.SMPL_param.body_pose.weight[0:1]),
            "global_orient": torch.zeros_like(model.SMPL_param.global_orient.weight[0:1]),
            "transl": torch.zeros_like(model.SMPL_param.transl.weight[0:1])
        }
        model.deformer.prepare_deformer(t_pose_params)
        
        # 2. Get template SMPL mesh for guidance
        smpl_verts = model.deformer.vs_template  # [1, N, 3]
        # Convert faces to torch tensor with correct dtype
        smpl_faces_np = model.deformer.body_model.faces.astype(np.int64)  # Convert to int64 first
        smpl_faces = torch.from_numpy(smpl_faces_np).to(device=smpl_verts.device)  # [F, 3]
        
        print(f"SMPL vertices shape: {smpl_verts.shape}") # [1, N, 3]
        print(f"SMPL faces shape: {smpl_faces.shape}") # [F, 3]
        
        # Prepare face vertices for distance computation
        mesh_face_vertices = smpl_verts[:, smpl_faces].clone()  # [1, F, 3, 3]
        print(f"Mesh face vertices shape: {mesh_face_vertices.shape}")
        
        def query_hybrid_field(points):
            """Query density field in canonical space"""
            with torch.no_grad():
                # Debug boundary points
                print(f"\nPoints range:")
                print(f"Min: {points.min(dim=0)[0]}")
                print(f"Max: {points.max(dim=0)[0]}")
                
                points_canonical, valid = model.deformer.deform(points, eval_mode=True)
                print(f"Points canonical shape: {points_canonical.shape}")
                print(f"Valid shape: {valid.shape}, Valid sum: {valid.sum()}")
                
                # Initialize field values
                field = torch.ones_like(points[..., 0]) * 1000.0  # [N]
                
                if valid.any():
                    # Get valid points mask
                    valid_points = valid.any(dim=-1)  # [N]
                    
                    # Get NeRF density for valid points
                    with torch.cuda.amp.autocast():
                        _, sigma = model.net_coarse(points_canonical[valid], None)
                    density = torch.zeros_like(valid, dtype=sigma.dtype)  # [N, 13]
                    density[valid] = sigma.reshape(-1)
                    
                    # Take the density at maximum sigma point for valid points
                    max_indices = density[valid_points].argmax(dim=-1)  # [V]
                    density_valid = density[valid_points][torch.arange(valid_points.sum()), max_indices]  # [V]
                    
                    # Get corresponding canonical points for valid points
                    points_valid = points_canonical[valid_points][torch.arange(valid_points.sum()), max_indices]
                    points_valid = points_valid.reshape(1, -1, 3).contiguous()  # [1, V, 3]
                    
                    # Compute SDF to SMPL template
                    dist = kaolin.metrics.trianglemesh.point_to_mesh_distance(
                        points_valid,  # [1, V, 3]
                        mesh_face_vertices  # [1, F, 3, 3]
                    )[0].sqrt()
                    
                    sign = kaolin.ops.mesh.check_sign(
                        smpl_verts,  # [1, V, 3] 
                        smpl_faces,  # [F, 3]
                        points_valid  # [1, V, 3]
                    ).float()
                    sign = 1 - 2 * sign
                    
                    # Create field values for valid points
                    sdf = sign * dist  # [V]
                    field_valid = sdf * torch.sigmoid(-density_valid) + \
                               (-torch.log(1 + density_valid)) * torch.sigmoid(density_valid)
                    
                    # Assign back to full field tensor
                    field[valid_points] = field_valid
                
                return field
        
        # 3. Multi-resolution extraction
        bbox = model.deformer.get_bbox_deformed()  # Returns [min_vals, max_vals]
        bbox = torch.stack([
            bbox[0].to("cuda"),  # min values
            bbox[1].to("cuda")   # max values
        ])
        print(f"Bbox tensor shape: {bbox.shape}, min: {bbox[0]}, max: {bbox[1]}")
        print(f"Using level_set: {level_set}")  # Debug print
        
        # Calculate resolution based on bbox size
        bbox_size = bbox[1] - bbox[0]
        max_size = bbox_size.max().item()
        max_resolution = int(base_resolution * max_size)  # Scale with model size
        
        # Scale resolutions proportionally
        resolutions = [
            max_resolution // 4,  # 128 if base_resolution is 512
            max_resolution // 2,  # 256
            max_resolution       # 512
        ]
        
        print(f"Using resolutions: {resolutions}")
        print(f"Using level_set: {level_set}")
        
        meshes = []
        for res in resolutions:
            print(f"\nProcessing resolution: {res}")
            try:
                mesh = marching_cubes(
                    func=query_hybrid_field,
                    bbox=bbox,
                    resolution=res,  # Use calculated resolution
                    level_set=level_set,
                    gradient_direction="ascent",
                    extract_max_component=True,
                    device="cuda"
                )
                print(f"Mesh vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}")
                meshes.append(mesh)
            except Exception as e:
                print(f"Error at resolution {res}: {e}")
                continue
        
        if not meshes:
            raise RuntimeError("Failed to extract mesh at any resolution")
            
        # 4. Return highest resolution result if multi-resolution failed
        if len(meshes) == 1:
            return meshes[0]
            
        # Convert trimesh vertices to torch tensors for merging
        final_vertices = torch.zeros(
            (len(meshes[-1].vertices), 3),
            dtype=torch.float32,
            device="cuda"
        )
        
        # Add debug prints for mesh vertices
        for i, mesh in enumerate(meshes):
            print(f"Mesh {i} vertices shape: {mesh.vertices.shape}")
            print(f"Mesh {i} faces shape: {mesh.faces.shape}")
        
        # Debug final_vertices initialization
        print(f"Final vertices shape: {final_vertices.shape}")
        
        weights = [0.2, 0.3, 0.5]  # Adjusted weights for three resolutions
        for i, (mesh, w) in enumerate(zip(meshes, weights)):
            source_verts = torch.from_numpy(
                np.array(mesh.vertices)
            ).float().cuda()
            target_verts = torch.from_numpy(
                np.array(meshes[-1].vertices)
            ).float().cuda()
            
            print(f"\nIteration {i}:")
            print(f"Source vertices shape: {source_verts.shape}")
            print(f"Target vertices shape: {target_verts.shape}")
            
            aligned_verts = interpolate_vertices(source_verts, target_verts)
            print(f"Aligned vertices shape: {aligned_verts.shape}")
            print(f"Weight value: {w}")
            print(f"Weight * aligned shape: {(aligned_verts * w).shape}")
            final_vertices += aligned_verts * w
        
        # Create final mesh with merged vertices
        final_mesh = trimesh.Trimesh(
            vertices=final_vertices.cpu().numpy(),
            faces=meshes[-1].faces
        )
        
        return final_mesh

@hydra.main(config_path="./confs", config_name="SNARF_NGP")
def main(opt):
    # Load trained model
    datamodule = hydra.utils.instantiate(opt.dataset, _recursive_=False)
    model = hydra.utils.instantiate(opt.model, datamodule=datamodule, _recursive_=False)
    model = model.cuda()
    model.eval()

    # Load checkpoint
    checkpoints = sorted(glob.glob("checkpoints/*.ckpt"))
    print("Loading checkpoint:", checkpoints[-1])
    checkpoint = torch.load(checkpoints[-1])
    model.load_state_dict(checkpoint["state_dict"])

    # Test SMPL parameter shapes
    test_smpl_params(model)

    # Define query function for marching cubes
    def query_func(points):
        with torch.no_grad():
            print(f"Input points shape: {points.shape}")
            # Take the first frame's parameters (index 0)
            body_pose = model.SMPL_param.body_pose.weight[0]  # Shape: [69]
            global_orient = model.SMPL_param.global_orient.weight[0]  # Shape: [3]
            
            # Reshape pose parameters to match SMPL expectations
            body_pose = body_pose.reshape(1, -1)  # [1, 69]
            global_orient = global_orient.reshape(1, -1)  # [1, 3]
            
            # Prepare deformer with current SMPL parameters
            smpl_params = {
                "betas": model.SMPL_param.betas.weight,  # Already [1, 10]
                "body_pose": body_pose,  # [1, 69]
                "global_orient": global_orient,  # [1, 3]
                "transl": model.SMPL_param.transl.weight[0:1]  # Take first frame, shape: [1, 3]
            }
            
            # Get canonical points using SNARF deformer
            model.deformer.prepare_deformer(smpl_params)
            points_canonical, valid = model.deformer.deform(points, eval_mode=True)
            
            # Debug points_canonical shape
            print(f"Points canonical shape: {points_canonical.shape}")
            print(f"Valid shape: {valid.shape}, Valid sum: {valid.sum()}")
            
            # Initialize density for all points
            density = torch.zeros_like(points[..., 0])  # Shape: [N]
            print(f"Initial density shape: {density.shape}")
            
            if valid.any():
                with torch.cuda.amp.autocast():
                    rgb, sigma = model.net_coarse(points_canonical[valid], None)
                    print(f"RGB shape: {rgb.shape}")
                    print(f"Sigma raw shape: {sigma.shape}")
                    print(f"Sigma min/max: {sigma.min():.3f}, {sigma.max():.3f}")
                    
                    # Reshape sigma to match valid points shape
                    sigma = sigma.reshape(valid.sum())
                    
                    # For each point, find the maximum sigma value among its valid candidates
                    sigma_reshaped = torch.zeros_like(valid, dtype=sigma.dtype)
                    sigma_reshaped[valid] = sigma
                    density = sigma_reshaped.mean(dim=-1)  # Use mean instead of max
                    
                    # Normalize density values, keep positive
                    density = torch.nn.functional.softplus(density)
            
            print(f"Final density shape: {density.shape}")
            return density

    # Get bounding box and density threshold
    bbox = get_smpl_bounds(model)
    level_set = get_density_stats(model, bbox, query_func)  # Pass query_func here
    
    print("Extracting mesh in canonical space...")
    mesh = extract_canonical_mesh(
        model, 
        level_set=level_set,
        base_resolution=256  # Can be adjusted if needed
    )
    
    # Save the mesh
    output_dir = Path(hydra.utils.to_absolute_path(opt.dataset.opt.dataroot)) / "nerf_mesh"
    output_dir.mkdir(exist_ok=True)
    
    mesh_path = output_dir / "detailed_mesh.obj"
    print(f"Saving mesh to {mesh_path}")
    mesh.export(str(mesh_path))

def test_smpl_params(model):
    print("SMPL Parameter Shapes:")
    print(f"betas: {model.SMPL_param.betas.weight.shape}")
    print(f"body_pose: {model.SMPL_param.body_pose.weight.shape}")
    print(f"global_orient: {model.SMPL_param.global_orient.weight.shape}")
    print(f"transl: {model.SMPL_param.transl.weight.shape}")
    
    print("\nFirst frame parameters:")
    body_pose = model.SMPL_param.body_pose.weight[0]  # First frame
    global_orient = model.SMPL_param.global_orient.weight[0]
    print(f"body_pose (first frame): {body_pose.shape}")  # Should be [69]
    print(f"global_orient (first frame): {global_orient.shape}")  # Should be [3]
    
    # Test conversion to rotation matrices
    body_pose_mat = axis_angle_to_matrix(
        body_pose.reshape(-1, 3)  # [23, 3]
    ).reshape(1, 23, 3, 3)
    print(f"\nAfter conversion to rotation matrices:")
    print(f"body_pose_mat: {body_pose_mat.shape}")  # Should be [1, 23, 3, 3]

if __name__ == "__main__":
    main()
