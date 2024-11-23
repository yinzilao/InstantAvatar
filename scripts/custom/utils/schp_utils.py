import os
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import subprocess
import cv2
from enum import Enum
from typing import List, OrderedDict
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add SCHP to path
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(f"PROJ_ROOT: {PROJ_ROOT}")

THIRD_PARTY_ROOT = Path(PROJ_ROOT) / "third_parties"
SCHP_PATH = Path(THIRD_PARTY_ROOT) / "schp"

if THIRD_PARTY_ROOT not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_ROOT))
print(f"sys.path: {sys.path}")

# Import SCHP modules
from schp.simple_extractor import dataset_settings  # From simple_extractor.py lines 27-46
import schp.networks as networks  # SCHP networks module
from schp.datasets.simple_extractor_dataset import SimpleFolderDataset
from schp.utils.transforms import transform_logits

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

def get_mask_for_labels(seg_map, model_type: ModelType, target_labels: set):
    """Create binary mask for specified labels"""
    is_tensor = torch.is_tensor(seg_map)
    if is_tensor:
        seg_map = seg_map.cpu().numpy()
        
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

def get_hair_hat_mask(seg_map, model_type: ModelType = ModelType.ATR):
    """Get binary mask for hair and hat regions"""
    is_tensor = torch.is_tensor(seg_map)
    
    if model_type != ModelType.PASCAL:
        hair_hat_labels = {'Hair', 'Hat'}
    else:
        raise ValueError("PASCAL model does not have hair or hat labels...")
    
    mask = get_mask_for_labels(seg_map, model_type, hair_hat_labels)
    
    # Convert to tensor if input was tensor
    if is_tensor:
        return torch.from_numpy(mask).to(seg_map.device)
    return mask

def get_face_mask(seg_map, model_type: ModelType = ModelType.ATR):
    """Get binary mask for face region"""
    is_tensor = torch.is_tensor(seg_map)
    
    if model_type != ModelType.PASCAL:
        face_label = {'Face'}
    else:
        raise ValueError("PASCAL model does not have face labels, "
                         "please use get_head_mask instead, "
                         "or you can use model ATR (recommended) to get face mask")
    
    mask = get_mask_for_labels(seg_map, model_type, face_label)
    
    # Convert to tensor if input was tensor
    if is_tensor:
        return torch.from_numpy(mask).to(seg_map.device)
    return mask

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

class SCHPModel:
    def __init__(self, model_type: ModelType, device="cuda"):
        self.model_type = model_type
        self.device = device
        
        # Get settings from SCHP's dataset_settings
        settings = dataset_settings[model_type.value]
        self.input_size = settings['input_size']
        self.num_classes = settings['num_classes']
        
        # Initialize model using SCHP's network initialization
        self.model = networks.init_model('resnet101', 
                                       num_classes=self.num_classes, 
                                       pretrained=None)
        
        # Load pretrained weights
        model_path = Path(SCHP_PATH) / "pretrained_models" / MODEL_PATHS[model_type]
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
        self.model.load_state_dict(new_state_dict)
        self.model.to(device)
        self.model.eval()
        
        # Setup transforms using same normalization as simple_extractor.py (lines 116-119)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], 
                              std=[0.225, 0.224, 0.229])
        ])

    @torch.no_grad()
    def process_batch(self, image_paths: List[str], batch_size: int = 12) -> List[torch.Tensor]:
        """
        Process images in batches and return segmentation maps
        
        Args:
            image_paths: List of paths to images
            batch_size: Batch size for processing
            
        Returns:
            List of segmentation maps as tensors on GPU
        """
        # Get the directory from the first image path
        root_dir = os.path.dirname(image_paths[0])
        
        dataset = SimpleFolderDataset(
            root=root_dir,
            input_size=self.input_size,
            transform=self.transform,
            file_list=[os.path.basename(p) for p in image_paths]  # Pass just the filenames
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True
        )
        
        seg_maps = []
        for batch in tqdm(dataloader):
            images, meta = batch
            images = images.to(self.device)
            
            outputs = self.model(images)
            
            for idx in range(images.size(0)):
                # Get metadata for this image
                c = meta['center'][idx].numpy()
                s = meta['scale'][idx].numpy()
                w = meta['width'][idx].numpy()
                h = meta['height'][idx].numpy()
                
                # Process output
                output = outputs[0][-1][idx].unsqueeze(0)
                upsample = torch.nn.Upsample(
                    size=self.input_size, 
                    mode='bilinear', 
                    align_corners=True
                )
                upsample_output = upsample(output)
                upsample_output = upsample_output.squeeze().permute(1, 2, 0)
                
                # Transform logits
                logits = torch.from_numpy(
                    transform_logits(
                        upsample_output.cpu().numpy(),
                        c, s, w, h,
                        input_size=self.input_size
                    )
                ).to(self.device)
                
                seg_map = torch.argmax(logits, dim=2)
                seg_maps.append(seg_map)
            
            # Optional: clear cache between batches
            torch.cuda.empty_cache()
            
        return seg_maps

def get_schp_segmentation_batch(model_type: ModelType, image_paths: list, batch_size: int = 12):
    """Process images in batches using GPU acceleration"""
    # Initialize model (or get from cache)
    model = SCHPModel(model_type)
    
    # Process all images in batches
    seg_maps = model.process_batch(image_paths, batch_size)
    
    return seg_maps

def main():
    model_type = ModelType.PASCAL
    input_image_path = "data/custom/e1/raw_images/00001.png"
    output_dir = Path(f"data/custom/e1/test_schp/{model_type.value}")
    
    # Get segmentation map
    seg_map = get_schp_segmentation(model_type, input_image_path)
    
    # Get different masks
    head_mask = get_head_mask(seg_map, model_type)
    hair_hat_mask = get_hair_hat_mask(seg_map, model_type)
    
    # Save results if needed
    os.makedirs(output_dir / "head_masks", exist_ok=True)
    os.makedirs(output_dir / "hair_hat_masks", exist_ok=True)
    os.makedirs(output_dir / "debug", exist_ok=True)
    
    Image.fromarray(head_mask.astype(np.uint8)).save(
        output_dir / "head_masks" / f"{os.path.splitext(os.path.basename(input_image_path))[0]}_head_mask.png"
    )
    Image.fromarray(hair_hat_mask.astype(np.uint8)).save(
        output_dir / "hair_hat_masks" / f"{os.path.splitext(os.path.basename(input_image_path))[0]}_hair_hat_mask.png"
    )
    
    # Save debug visualization
    orig_img = cv2.imread(input_image_path)
    if orig_img is not None:
        save_debug_visualization(
            seg_map, orig_img, model_type,
            output_dir / "debug" / f"{os.path.splitext(os.path.basename(input_image_path))[0]}_parsed.png"
        )
        save_debug_visualization(
            head_mask, orig_img, model_type,
            output_dir / "debug" / f"{os.path.splitext(os.path.basename(input_image_path))[0]}_head_mask.png"
        )
        save_debug_visualization(
            hair_hat_mask, orig_img, model_type,
            output_dir / "debug" / f"{os.path.splitext(os.path.basename(input_image_path))[0]}_hair_hat_mask.png"
        )

if __name__ == "__main__":
    main() 