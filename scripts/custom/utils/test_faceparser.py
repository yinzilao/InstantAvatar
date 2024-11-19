import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import tempfile

# Add the third_parties/face-parsing.PyTorch directory to Python path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
face_parsing_path = os.path.join(project_root, 'third_parties/face-parsing.PyTorch')
sys.path.append(face_parsing_path)

from model import BiSeNet  # You'll need to have the model.py file from the repo

class FaceParser:
    def __init__(self, model_path='79999_iter.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Add warning about PyTorch version
        print("Note: Using PyTorch version:", torch.__version__)
        try:
            self.model = self._load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Try downloading the ResNet18 weights manually and placing them in ~/.cache/torch/hub/checkpoints/")
            raise
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
    def _load_model(self, model_path):
        try:
            # Initialize ResNet18 without pretrained weights
            import torchvision.models as models
            resnet = models.resnet18(weights=None)
            
            # Download weights manually with better error handling
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # First handle ResNet18 weights
            weights_url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
            
            # Configure retries
            session = requests.Session()
            retries = Retry(total=5, backoff_factor=0.1)
            session.mount('https://', HTTPAdapter(max_retries=retries))
            
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
                print(f"Downloading ResNet18 weights from {weights_url}")
                response = session.get(weights_url, stream=True)
                response.raise_for_status()
                
                with open(tmp.name, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                print("Successfully downloaded weights, loading into model...")
                state_dict = torch.load(tmp.name, map_location=self.device)
                resnet.load_state_dict(state_dict)
                print("Successfully loaded ResNet18 weights")
                
                os.unlink(tmp.name)
            
            # Modify the Resnet18 class in the face parsing model
            from resnet import Resnet18
            Resnet18.init_weight = lambda self: setattr(self, 'state_dict', resnet.state_dict())
            
        except Exception as e:
            print(f"Error loading ResNet18: {str(e)}")
            raise

        # Now handle face parsing model weights
        if not os.path.exists(model_path):
            print(f"Downloading face parsing model weights to {model_path}")
            face_weights_url = "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812"
            
            try:
                import gdown
                gdown.download(face_weights_url, model_path, quiet=False)
            except Exception as e:
                raise FileNotFoundError(f"Failed to download face parsing weights: {str(e)}. Please download manually from the repository and place at {model_path}")

        model = BiSeNet(n_classes=19)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def parse(self, image_path, save_visualization=True):
        """
        Parse face image and return hair mask
        Args:
            image_path: path to image file
            save_visualization: whether to save visualization of the hair mask
        Returns:
            hair_mask: binary mask of hair region
        """
        # Read image
        img = Image.open(image_path).convert('RGB')
        orig_size = img.size
        
        # Preprocess
        input_tensor = self.transform(img)
        input_tensor = torch.unsqueeze(input_tensor, 0)
        input_tensor = input_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            out = self.model(input_tensor)[0]
            parsing = out.squeeze(0).argmax(0).cpu().numpy()
        
        # Get hair mask (class 17 is hair in this model)
        hair_mask = (parsing == 17).astype(np.uint8)
        
        # Resize back to original size
        hair_mask = cv2.resize(hair_mask, (orig_size[0], orig_size[1]), 
                             interpolation=cv2.INTER_NEAREST)
        
        if save_visualization:
            # Save visualization
            vis_path = image_path.rsplit('.', 1)[0] + '_hair_mask.png'
            orig_img = cv2.imread(image_path)
            # Create colored overlay
            vis_img = orig_img.copy()
            vis_img[hair_mask == 1] = [0, 255, 0]  # Green overlay for hair
            # Blend with original
            vis_img = cv2.addWeighted(orig_img, 0.7, vis_img, 0.3, 0)
            cv2.imwrite(vis_path, vis_img)
            
            # Save binary mask
            mask_path = image_path.rsplit('.', 1)[0] + '_hair_binary.png'
            cv2.imwrite(mask_path, hair_mask * 255)
            
        return hair_mask

def test_on_folder(input_folder, model_path='79999_iter.pth'):
    """
    Test face parsing on all images in a folder
    """
    parser = FaceParser(model_path)
    
    # Process all images in folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")
            try:
                hair_mask = parser.parse(image_path)
                print(f"Successfully processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    input_folder = "data/custom/e1/head_bbox_crops/"
    model_path = "79999_iter.pth"  # Download this from the repository
    
    test_on_folder(input_folder, model_path)
