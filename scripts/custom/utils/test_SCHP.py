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

# Import labels for ATR dataset
ATR_LABELS = ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 
              'Dress', 'Belt', 'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 
              'Left-arm', 'Right-arm', 'Bag', 'Scarf']

def setup_schp():
    """Initialize SCHP model with ATR dataset weights"""
    dataset = 'atr'
    model_path = SCHP_PATH / "pretrained_models/exp-schp-201908301523-atr.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please download it from: "
            "https://drive.google.com/file/d/1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP/view"
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
        "--output-dir", output_dir,
        "--logits"
    ]
    
    subprocess.run(cmd, check=True)
    return output_dir

def create_head_masks(parsing_dir, output_dir, debug_dir):
    """Create binary masks for head regions from parsing results"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    # Get indices for head-related labels using ATR labels
    head_label_names = {'Hat', 'Hair', 'Face'}
    head_labels = {ATR_LABELS.index(label) for label in head_label_names}
    
    # Create color map for all labels
    np.random.seed(42)
    all_label_colors = {
        i: tuple(map(int, np.random.randint(0, 255, 3)))
        for i in range(len(ATR_LABELS))
    }
    
    # Special colors for head parts
    all_label_colors[ATR_LABELS.index('Hat')] = (255, 0, 0)    # Red
    all_label_colors[ATR_LABELS.index('Hair')] = (0, 255, 0)   # Green
    all_label_colors[ATR_LABELS.index('Face')] = (0, 0, 255)   # Blue
    
    for parsing_file in os.listdir(parsing_dir):
        if not parsing_file.endswith('.png'):
            continue
            
        # Load parsing result
        parsing_path = os.path.join(parsing_dir, parsing_file)
        parsing = np.array(Image.open(parsing_path))
        
        # Create visualization with all labels
        orig_img = cv2.imread(f"data/custom/e1/raw_images/{os.path.splitext(parsing_file)[0]}.png")
        if orig_img is not None:
            # Create full parsing visualization
            parsing_vis = np.zeros_like(orig_img)
            for label_idx, color in all_label_colors.items():
                parsing_vis[parsing == label_idx] = color
            
            # Blend original image with parsing
            debug_img = cv2.addWeighted(orig_img, 0.7, parsing_vis, 0.3, 0)
            
            # Add label names in the center of each segment
            for label_idx, label_name in enumerate(ATR_LABELS):
                # Get mask for current label
                mask = parsing == label_idx
                if not np.any(mask):
                    continue
                
                # Find contours of the segment
                mask = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # For each contour (segment) of this label
                for contour in contours:
                    if cv2.contourArea(contour) < 100:  # Skip tiny segments
                        continue
                    
                    # Calculate center of mass
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Draw arrow from outside the segment to its center
                        color = all_label_colors[label_idx]
                        
                        # Get text size for better positioning
                        text_size = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        
                        # Draw text with background for better visibility
                        cv2.putText(debug_img, label_name, 
                                  (cx - text_size[0]//2, cy),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # thick black outline
                        cv2.putText(debug_img, label_name, 
                                  (cx - text_size[0]//2, cy),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # colored text
            
            cv2.imwrite(os.path.join(debug_dir, f"{os.path.splitext(parsing_file)[0]}_parsed.png"), 
                       debug_img)
        
        # Create and save head mask
        head_mask = np.zeros_like(parsing)
        for label in head_labels:
            head_mask[parsing == label] = 255
        output_path = os.path.join(output_dir, f"{os.path.splitext(parsing_file)[0]}_head_mask.png")
        Image.fromarray(head_mask.astype(np.uint8)).save(output_path)

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