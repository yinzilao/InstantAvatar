import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

def preprocess_image(image):
    # Convert to LAB color space which separates luminance from color
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge channels
    limg = cv2.merge((cl,a,b))
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Additional denoising
    denoised = cv2.fastNlMeansDenoisingColored(enhanced)
    
    # 1. Bilateral Filter for edge-preserving smoothing
    bilateral = cv2.bilateralFilter(denoised, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 2. Gamma correction to enhance darker regions
    gamma = 1.2
    gamma_corrected = np.power(bilateral/255.0, gamma) * 255.0
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    # 3. Sharpening using unsharp masking
    gaussian = cv2.GaussianBlur(gamma_corrected, (0, 0), 3.0)
    sharpened = cv2.addWeighted(gamma_corrected, 1.5, gaussian, -0.5, 0)
    
    return sharpened

def main():
    parser = argparse.ArgumentParser(description='Preprocess images in a directory')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--image_processed_dir', type=str, required=True, help='Name of the processed image output folder')
    args = parser.parse_args()

    # Create output directory
    output_dir = args.image_processed_dir
    os.makedirs(output_dir, exist_ok=True)

    # Process all images in the directory
    image_files = [f for f in os.listdir(args.data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files, desc="Preprocessing images"):
        # Read image
        image_path = os.path.join(args.data_dir, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Warning: Could not read image {image_file}")
            continue
            
        # Process image
        processed = preprocess_image(image)
        
        # Save processed image
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, processed)

if __name__ == "__main__":
    main()