import os
import argparse
import cv2
import numpy as np
from pathlib import Path

from utils.schp_wrapper import ModelType, setup_schp, create_head_masks

def create_body_only_masks(sam_masks_dir, head_masks_dir, body_only_masks_dir):
    """Subtract head masks from SAM masks to create body-only masks"""
    os.makedirs(body_only_masks_dir, exist_ok=True)
    
    for mask_file in os.listdir(sam_masks_dir):
        if not mask_file.endswith('.png'):
            continue
            
        # Load SAM mask and head mask
        sam_mask = cv2.imread(os.path.join(sam_masks_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        head_mask_file = f"{os.path.splitext(mask_file)[0]}_head_mask.png"
        head_mask = cv2.imread(os.path.join(head_masks_dir, head_mask_file), cv2.IMREAD_GRAYSCALE)
        
        if head_mask is None:
            print(f"Warning: Head mask not found for {mask_file}")
            continue
        
        # Improve head mask coverage using morphological operations
        improved_head_mask = improve_head_masks(head_mask)
        
        # Create body-only mask by subtracting improved head mask from SAM mask
        body_only_mask = sam_mask.copy()
        body_only_mask[improved_head_mask > 127] = 0
        
        # Save body-only mask
        cv2.imwrite(os.path.join(body_only_masks_dir, mask_file), body_only_mask)

def create_masked_images(image_dir, mask_dir, output_dir):
    """Create masked images using body-only masks"""
    os.makedirs(output_dir, exist_ok=True)
    
    for mask_file in os.listdir(mask_dir):
        if not mask_file.endswith('.png'):
            continue
            
        # Load image and mask
        image_file = f"{os.path.splitext(mask_file)[0]}.png"
        image = cv2.imread(os.path.join(image_dir, image_file))
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Warning: Image not found for {mask_file}")
            continue
        
        # Apply mask
        masked_image = image.copy()
        masked_image[mask == 0] = 0
        
        # Save masked image
        cv2.imwrite(os.path.join(output_dir, image_file), masked_image)

def improve_head_masks(head_mask):
    """Apply morphological operations to improve head mask coverage"""
    # Create elliptical kernels for more natural head shape operations
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # 1. Initial dilation to catch thin edges of hair and chin
    dilated_mask = cv2.dilate(head_mask, kernel_dilate, iterations=2)
    
    # 2. Smooth the edges
    smoothed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel_smooth)
    
    # 3. Final slight dilation to ensure coverage
    final_mask = cv2.dilate(smoothed_mask, kernel_smooth, iterations=1)
    
    return final_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory containing the data')
    parser.add_argument('--image_folder', required=True, help='Folder containing input images')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    image_dir = data_dir / args.image_folder
    sam_masks_dir = data_dir / "masks_sam"
    
    # Setup paths
    schp_dir = data_dir / "debug_schp"
    head_masks_dir = schp_dir / "head_masks"
    body_only_masks_dir = data_dir / "body_only_masks_schp"
    body_only_masked_images_dir = data_dir / "body_only_masked_images_schp"
    
    # Run SCHP to get head masks
    print("Running SCHP for head segmentation...")
    parsing_dir = setup_schp(ModelType.PASCAL, schp_dir)
    create_head_masks(ModelType.PASCAL, parsing_dir, head_masks_dir, schp_dir)
    
    # Create body-only masks
    print("Creating body-only masks...")
    create_body_only_masks(sam_masks_dir, head_masks_dir, body_only_masks_dir)
    
    # Create masked images
    print("Creating body-only masked images...")
    create_masked_images(image_dir, body_only_masks_dir, body_only_masked_images_dir)
    
    print("Processing complete!")

if __name__ == "__main__":
    main() 