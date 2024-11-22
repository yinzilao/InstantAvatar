import os
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import subprocess
import cv2
from enum import Enum

# Add SCHP to path
SCHP_PATH = Path("third_parties/Self-Correction-Human-Parsing")
sys.path.append(str(SCHP_PATH))

class ModelType(Enum):
    LIP = 'lip'
    ATR = 'atr'
    PASCAL = 'pascal'

# Define labels for each model
LABELS_BY_MODEL = {
    ModelType.LIP: [
        'Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 
        'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 
        'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 
        'Right-shoe'
    ],
    ModelType.ATR: [
        'Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 
        'Pants', 'Dress', 'Belt', 'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 
        'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf'
    ],
    ModelType.PASCAL: [
        'Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 
        'Lower Legs'
    ]
}

MODEL_PATHS = {
    ModelType.LIP: "exp-schp-201908261155-lip.pth",
    ModelType.ATR: "exp-schp-201908301523-atr.pth",
    ModelType.PASCAL: "exp-schp-201908270938-pascal-person-part.pth"
}

# Define head-related labels for each model
HEAD_LABELS_BY_MODEL = {
    ModelType.LIP: {'Hat', 'Hair', 'Face'},  # From README.md line 34
    ModelType.ATR: {'Hat', 'Hair', 'Face'},  # From README.md line 40
    ModelType.PASCAL: {'Head'}               # From README.md line 46
}

def get_schp_segmentation(model_type: ModelType, input_image_path: str):
    """Get segmentation map for a single image"""
    model_path = SCHP_PATH / "pretrained_models" / MODEL_PATHS[model_type]
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please download it from SCHP repository."
        )
    
    # Run SCHP inference and return segmentation map
    cmd = [
        "python", 
        str(SCHP_PATH / "simple_extractor.py"),
        "--dataset", model_type.value,
        "--model-restore", str(model_path),
        "--input-dir", os.path.dirname(input_image_path),
        "--output-dir", "temp_output",
        "--logits"
    ]
    
    subprocess.run(cmd, check=True)
    seg_map = np.array(Image.open(f"temp_output/{os.path.basename(input_image_path)}"))
    return seg_map

def get_mask_for_labels(seg_map: np.ndarray, model_type: ModelType, target_labels: set):
    """Create binary mask for specified labels"""
    labels = LABELS_BY_MODEL[model_type]
    label_indices = {labels.index(label) for label in target_labels if label in labels}
    
    mask = np.zeros_like(seg_map)
    for label in label_indices:
        mask[seg_map == label] = 255
    return mask

def get_head_mask(seg_map: np.ndarray, model_type: ModelType):
    """Get binary mask for head region"""
    head_label_names = HEAD_LABELS_BY_MODEL[model_type]
    return get_mask_for_labels(seg_map, model_type, head_label_names)

def get_hair_hat_mask(seg_map: np.ndarray, model_type: ModelType):
    """Get binary mask for hair and hat regions"""
    hair_hat_labels = {'Hair', 'Hat'} if model_type != ModelType.PASCAL else {'Head'}
    return get_mask_for_labels(seg_map, model_type, hair_hat_labels)

def save_debug_visualization(seg_map: np.ndarray, orig_img: np.ndarray, 
                           model_type: ModelType, output_path: str):
    """Save debug visualization with labels and colors"""
    labels = LABELS_BY_MODEL[model_type]
    
    # Create color map
    np.random.seed(42)
    all_label_colors = {
        i: tuple(map(int, np.random.randint(0, 255, 3)))
        for i in range(len(labels))
    }
    
    # Set special colors for head parts
    if model_type == ModelType.PASCAL:
        all_label_colors[labels.index('Head')] = (0, 255, 0)
    else:
        if 'Hat' in labels:
            all_label_colors[labels.index('Hat')] = (255, 0, 0)
        if 'Hair' in labels:
            all_label_colors[labels.index('Hair')] = (0, 255, 0)
        if 'Face' in labels:
            all_label_colors[labels.index('Face')] = (0, 0, 255)
    
    # Create visualization
    parsing_vis = np.zeros_like(orig_img)
    for label_idx, color in all_label_colors.items():
        parsing_vis[seg_map == label_idx] = color
    
    debug_img = cv2.addWeighted(orig_img, 0.7, parsing_vis, 0.3, 0)
    
    # Add label annotations
    for label_idx, label_name in enumerate(labels):
        mask = seg_map == label_idx
        if not np.any(mask):
            continue
        
        mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue
            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                color = all_label_colors[label_idx]
                text_size = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                cv2.putText(debug_img, label_name, 
                          (cx - text_size[0]//2, cy),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(debug_img, label_name, 
                          (cx - text_size[0]//2, cy),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imwrite(output_path, debug_img)

def main():
    model_type = ModelType.PASCAL
    input_image_path = "data/custom/e1/raw_images/example.png"
    output_dir = Path(f"data/custom/e1/test_schp/{model_type.value}")
    
    # Get segmentation map
    seg_map = get_schp_segmentation(model_type, input_image_path)
    
    # Get different masks
    head_mask = get_head_mask(seg_map, model_type)
    hair_hat_mask = get_hair_hat_mask(seg_map, model_type)
    
    # Save results if needed
    os.makedirs(output_dir / "head_masks", exist_ok=True)
    os.makedirs(output_dir / "debug", exist_ok=True)
    
    Image.fromarray(head_mask.astype(np.uint8)).save(
        output_dir / "head_masks" / f"{os.path.splitext(os.path.basename(input_image_path))[0]}_head_mask.png"
    )
    
    # Save debug visualization
    orig_img = cv2.imread(input_image_path)
    if orig_img is not None:
        save_debug_visualization(
            seg_map, orig_img, model_type,
            output_dir / "debug" / f"{os.path.splitext(os.path.basename(input_image_path))[0]}_parsed.png"
        )

if __name__ == "__main__":
    main() 