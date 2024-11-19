import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
from collections import OrderedDict
import torch.nn.functional as F

# Add SOLIDER path to system path
SOLIDER_PATH = os.path.join(os.getcwd(), 'third_parties/SOLIDER')
sys.path.append(SOLIDER_PATH)

# Import SOLIDER's model
from swin_transformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224

def setup_test_folders(base_dir):
    """Setup folders for test outputs"""
    test_dir = os.path.join(base_dir, 'test_solider')  # Create test_solider subfolder
    os.makedirs(os.path.join(test_dir, 'parsing'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'final'), exist_ok=True)
    return test_dir

def load_solider_model():
    """Load SOLIDER model"""
    model = swin_small_patch4_window7_224(
        img_size=(256, 128),
        convert_weights=False,
        semantic_weight=1.0
    )
    
    checkpoint_path = os.path.join(SOLIDER_PATH, "checkpoints/swin_small.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Downloading SOLIDER checkpoint to {checkpoint_path}")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        import gdown
        url = "https://drive.google.com/uc?id=1oyEgASqDHc7YUPsQUMxuo2kBZyi2Tzfv"
        gdown.download(url, checkpoint_path, quiet=False)
    
    model.init_weights(checkpoint_path)
    return model.cuda()

def process_images(input_folder, output_folder):
    """Process images using SOLIDER for head/hair segmentation"""
    model = load_solider_model()
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 128)),  # Keep original aspect ratio
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    for img_name in os.listdir(input_folder):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        print(f"Processing {img_name}...")
        try:
            img_path = os.path.join(input_folder, img_name)
            img = Image.open(img_path).convert('RGB')
            original_size = img.size
            
            # Preserve aspect ratio while resizing
            img_tensor = transform(img).unsqueeze(0).cuda()
            
            with torch.no_grad():
                features, outs = model(img_tensor)  # Get both outputs
                
                # Process features for parsing
                parsing_features = F.interpolate(
                    outs[-1], 
                    size=(256, 128),  # Match input size
                    mode='bilinear', 
                    align_corners=True
                )
                parsing_pred = torch.argmax(parsing_features, dim=1)
                
                # Save parsing visualization at original resolution
                parsing_vis = visualize_parsing(parsing_pred[0].cpu().numpy())
                parsing_vis = cv2.resize(
                    parsing_vis, 
                    (original_size[0], original_size[1]),
                    interpolation=cv2.INTER_NEAREST
                )
                
                parsing_path = os.path.join(output_folder, 'parsing', f'{os.path.splitext(img_name)[0]}_parsing.png')
                cv2.imwrite(parsing_path, cv2.cvtColor(parsing_vis, cv2.COLOR_RGB2BGR))
                print(f"Saved parsing to: {parsing_path}")
                
                # Create and save masked image
                head_mask = get_head_mask(parsing_pred[0].cpu().numpy())
                final_result = apply_mask(img, head_mask, original_size)
                
                final_path = os.path.join(output_folder, 'final', f'{os.path.splitext(img_name)[0]}_final.png')
                cv2.imwrite(final_path, cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))
                print(f"Saved final result to: {final_path}")
                
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")

def visualize_parsing(parsing):
    """Visualize parsing results with colors"""
    vis = np.zeros((*parsing.shape, 3), dtype=np.uint8)
    # Define colors for different parts
    colors = {
        0: [255, 0, 0],    # head
        1: [0, 255, 0],    # hair
        2: [0, 0, 255],    # face
    }
    for label, color in colors.items():
        vis[parsing == label] = color
    return vis

def get_head_mask(parsing):
    """Extract head mask including hair"""
    mask = np.zeros_like(parsing, dtype=np.uint8)
    head_classes = [0, 1]  # head and hair classes
    for cls in head_classes:
        mask[parsing == cls] = 255
    return mask

def apply_mask(image, mask, original_size):
    """Apply mask to original image"""
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Resize mask to match original image size
    mask = cv2.resize(mask, (original_size[0], original_size[1]), 
                     interpolation=cv2.INTER_NEAREST)
    
    # Apply mask
    result = cv2.bitwise_and(image_np, image_np, mask=mask)
    return result

if __name__ == "__main__":
    input_folder = "data/custom/e1/raw_images"
    test_dir = setup_test_folders("data/custom/e1/test_solider")
    process_images(input_folder, test_dir)
