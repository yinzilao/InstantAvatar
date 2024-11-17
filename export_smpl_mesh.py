import hydra
import torch
import numpy as np
from pathlib import Path
import os
import glob

@hydra.main(config_path="./confs", config_name="SNARF_NGP_fitting")
def export_mesh(opt):
    print(f"Switch to {os.getcwd()}")
    
    # Initialize the same model setup as in fit.py
    datamodule = hydra.utils.instantiate(opt.dataset, _recursive_=False)
    model = hydra.utils.instantiate(opt.model, datamodule=datamodule, _recursive_=False)
    
    # Move model to GPU
    model = model.cuda()
    # Explicitly move SMPL model to GPU
    model.deformer.body_model = model.deformer.body_model.cuda()
    
    # Load the latest checkpoint
    checkpoints = sorted(glob.glob("checkpoints/fit/*.ckpt"))
    if len(checkpoints) > 0:
        print("Loading best model from", checkpoints[-1])
        checkpoint = torch.load(checkpoints[-1])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError("No checkpoint found to export mesh from!")

    # Create a dummy batch index (0) to get the first frame
    with torch.no_grad():
        # Get SMPL parameters through the embedding
        idx = torch.zeros(1, dtype=torch.long).cuda()
        body_params = model.SMPL_param(idx)
        
        # Create batch dictionary with all tensors on GPU
        batch = {
            "global_orient": body_params["global_orient"].cuda(),
            "body_pose": body_params["body_pose"].cuda(),
            "transl": body_params["transl"].cuda(),
            "betas": body_params["betas"].cuda()
        }
        
        # Initialize the deformer first
        model.deformer.initialize(batch["betas"], batch["betas"].device)
        
        # Get the mesh directly from SMPL model
        smpl_outputs = model.deformer.body_model(
            betas=batch["betas"],
            body_pose=batch["body_pose"],
            global_orient=batch["global_orient"],
            transl=batch["transl"]
        )
        
        # Get vertices and faces
        vertices = smpl_outputs.vertices.detach().cpu().numpy()[0]
        faces = model.deformer.body_model.faces  # faces are already numpy array
        
        # Save as OBJ file
        root = hydra.utils.to_absolute_path(opt.dataset.opt.dataroot)
        mesh_root = Path(root) / "smpl_meshes"
        mesh_root.mkdir(exist_ok=True)
        
        mesh_path = mesh_root / "optimized_smpl_mesh.obj"
        print(f"Saving mesh to {mesh_path}")
        
        with open(mesh_path, 'w') as f:
            for v in vertices:
                f.write(f'v {v[0]} {v[1]} {v[2]}\n')
            for face in faces:
                f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')

if __name__ == "__main__":
    export_mesh()
