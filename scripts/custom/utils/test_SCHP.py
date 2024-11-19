import os
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import subprocess
import cv2

# Add SCHP to path
SCHP_PATH = Path("third_parties/Self-Correction-Human-Parsing")
sys.path.append(str(SCHP_PATH))

# Import labels from miou.py
from utils.miou import LABELS, get_palette

def setup_schp():
    """Initialize SCHP model with LIP dataset weights"""
    dataset = 'lip'
    model_path = SCHP_PATH / "pretrained_models/exp-schp-201908261155-lip.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please download it from: "
            "https://drive.google.com/file/d/1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH/view"
        )
    
    input_dir = "data/custom/e1/raw_images"
    output_dir = "data/custom/e1/test_schp/parsing_results"
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "python", 
        str(SCHP_PATH / "simple_extractor.py"),
        "--dataset", dataset,
        "--model-restore", str(model_path),
        "--input-dir", input_dir,
        "--output-dir", output_dir
    ]
    
    subprocess.run(cmd, check=True)
    return output_dir

def create_head_masks(parsing_dir, output_dir, debug_dir):
    """Create binary masks for head regions from parsing results"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    # Get indices for head-related labels from LABELS list
    head_label_names = {'Hat', 'Hair', 'Face'}
    head_labels = {LABELS.index(label) for label in head_label_names}
    
    # Create a debug color map for visualization
    debug_colors = {
        LABELS.index('Hat'): (255, 0, 0),    # Red for hat
        LABELS.index('Hair'): (0, 255, 0),   # Green for hair
        LABELS.index('Face'): (0, 0, 255)    # Blue for face
    }
    
    for parsing_file in os.listdir(parsing_dir):
        if not parsing_file.endswith('.png'):
            continue
            
        # Load parsing result
        parsing_path = os.path.join(parsing_dir, parsing_file)
        parsing = np.array(Image.open(parsing_path))
        
        # Create visualization of full parsing result
        vis_parsing = Image.fromarray(parsing.astype(np.uint8))
        vis_parsing.putpalette(get_palette(len(LABELS)))
        vis_parsing.save(os.path.join(debug_dir, f"{os.path.splitext(parsing_file)[0]}_parsed.png"))
        
        # Create binary mask for head regions
        head_mask = np.zeros_like(parsing)
        for label in head_labels:
            head_mask[parsing == label] = 255
            
        # Save head mask
        output_path = os.path.join(output_dir, f"{os.path.splitext(parsing_file)[0]}_head_mask.png")
        Image.fromarray(head_mask.astype(np.uint8)).save(output_path)
        
        # Create colored debug visualization
        orig_img = cv2.imread(f"data/custom/e1/raw_images/{os.path.splitext(parsing_file)[0]}.png")
        if orig_img is not None:
            # Create colored overlay for different head parts
            overlay = np.zeros_like(orig_img)
            for label, color in debug_colors.items():
                overlay[parsing == label] = color
                
            # Blend original image with overlay
            debug_img = cv2.addWeighted(orig_img, 0.7, overlay, 0.3, 0)
            
            # Add legend
            y_offset = 30
            for label_name, color in zip(['Hat', 'Hair', 'Face'], [(255,0,0), (0,255,0), (0,0,255)]):
                cv2.putText(debug_img, label_name, (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                y_offset += 30
                
            cv2.imwrite(os.path.join(debug_dir, f"{os.path.splitext(parsing_file)[0]}_head_masked.png"), 
                       debug_img)

def main():
    output_dir = Path("data/custom/e1/test_schp")
    parsing_dir = setup_schp()
    head_masks_dir = output_dir / "head_masks"
    debug_dir = output_dir / "debug"
    
    print("Creating head masks and debug visualizations...")
    create_head_masks(parsing_dir, head_masks_dir, debug_dir)
    print("Processing complete!")

if __name__ == "__main__":
    main() 