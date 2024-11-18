import cv2
import numpy as np
from skimage import filters
from skimage import morphology
import os
from scipy import ndimage

class HairSegmenter:
    def __init__(self):
        self.color_threshold = 45
        self.texture_threshold = 0.4
        self.min_hair_area = 100  # Minimum area for connected components
        
    def get_validated_hair_mask(self, image, face_mask, head_bbox, confidence):
        """Hair segmentation with validation pipeline"""
        try:
            print("\nHair Segmentation Debug:")
            
            # 1. Get face color model
            face_color_model = self.get_face_color_model(image, face_mask)
            if face_color_model is None:
                print("Failed to get face color model")
                return None, {'error': "Failed to get face color model"}
            print(f"Face color model: mean={face_color_model['mean']}, std={face_color_model['std']}")
                
            # 2. Generate hair candidates
            hair_candidates = self.get_hair_candidates(
                image, face_color_model, head_bbox, confidence
            )
            if hair_candidates is None:
                print("No hair candidates found")
                return None, {'error': "No hair candidates found"}
            print(f"Found {np.sum(hair_candidates)} hair candidate pixels")
            
            # 3. Region growing
            hair_regions = self.grow_hair_regions(
                hair_candidates, face_mask, head_bbox
            )
            if hair_regions is None:
                print("Region growing failed")
                return None, {'error': "Region growing failed"}
            print(f"Hair regions after growing: {np.sum(hair_regions)} pixels")
            
            # 4. Final validation
            valid_hair_mask = self.validate_hair_regions(
                hair_regions, face_mask, head_bbox, confidence
            )
            if valid_hair_mask is None:
                print("Hair regions validation failed")
                return None, {'error': "Hair regions validation failed"}
            else:
                print(f"Final hair mask: {np.sum(valid_hair_mask)} pixels")
                
            # Instead of saving images, collect them in debug_data
            debug_data = {
                'candidates': hair_candidates,
                'regions': hair_regions,
                'final_mask': valid_hair_mask,
                'face_color_model': face_color_model,
                'debug_images': {
                    'hair_candidates': hair_candidates.astype(np.uint8) * 255 if hair_candidates is not None else None,
                    'hair_regions': hair_regions.astype(np.uint8) * 255 if hair_regions is not None else None,
                    'final_hair_mask': valid_hair_mask.astype(np.uint8) * 255 if valid_hair_mask is not None else None,
                    'hair_masked': image.copy() * valid_hair_mask[:,:,np.newaxis] if valid_hair_mask is not None else None
                }
            }
            
            return valid_hair_mask, debug_data
            
        except Exception as e:
            print(f"Error in hair segmentation: {e}")
            return None, {'error': str(e)}
            
    def get_face_color_model(self, image, face_mask):
        """Get color statistics from face region"""
        # Convert to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Get face region pixels
        face_pixels = lab_image[face_mask]
        if len(face_pixels) == 0:
            return None
            
        # Calculate color statistics
        mean = np.mean(face_pixels, axis=0)
        std = np.std(face_pixels, axis=0)
        
        return {'mean': mean, 'std': std}
        
    def get_adaptive_thresholds(self, image, bbox):
        """Calculate adaptive thresholds based on image characteristics"""
        # Crop to head region for local analysis
        y1, x1, y2, x2 = [int(x) for x in bbox]
        head_region = image[max(0, y1):y2, max(0, x1):x2]
        
        if head_region.size == 0:
            return None
        
        # Convert to HSV for analysis
        hsv_region = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        
        # Calculate local statistics
        mean_v = np.mean(hsv_region[:,:,2])
        std_v = np.std(hsv_region[:,:,2])
        mean_s = np.mean(hsv_region[:,:,1])
        
        # Adaptive thresholds based on image statistics
        thresholds = {
            'texture': {
                'base': 0.5,
                'scale': 1.0 + (std_v / 128.0)  # Adjust based on value contrast
            },
            'dark_hair': {
                'base': 0.7,
                'scale': mean_v / 128.0  # Darker threshold for bright images
            },
            'light_hair': {
                'base': 1.3,
                'scale': 2.0 - (mean_v / 128.0)  # Higher threshold for dark images
            },
            'saturation': {
                'base': 1.2,
                'scale': 1.0 + (mean_s / 128.0)  # Adjust based on color saturation
            }
        }
        
        return thresholds
        
    def get_hair_candidates(self, image, face_color_model, bbox, confidence):
        """Generate hair region candidates"""
        try:
            # Convert to multiple color spaces
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Clip bbox to image boundaries
            h, w = image.shape[:2]
            y1 = max(0, int(bbox[1]))
            x1 = max(0, int(bbox[0]))
            y2 = min(h, int(bbox[3]))
            x2 = min(w, int(bbox[2]))
            
            # Texture detection with stricter threshold
            gradient_x = filters.sobel_h(gray_image)
            gradient_y = filters.sobel_v(gray_image)
            gradient = np.sqrt(gradient_x**2 + gradient_y**2)
            texture_mask = gradient > np.percentile(gradient, 60)  # Increased threshold
            
            # Create spatial mask relative to clipped bbox
            spatial_mask = np.zeros_like(gray_image, dtype=bool)
            head_height = y2 - y1
            spatial_mask[
                max(0, y1):min(h, y1+int(head_height)),  # Only above face
                max(0, x1-int((x2-x1)*0.2)):min(w, x2+int((x2-x1)*0.2))
            ] = 1
            
            # Combine masks
            hair_candidates = texture_mask & spatial_mask
            
            return hair_candidates

        except Exception as e:
            print(f"Error in hair candidate generation: {e}")
            return None
        
    def grow_hair_regions(self, candidates, face_mask, bbox):
        """Grow regions from face boundary with improved coverage"""
        if candidates is None or face_mask is None:
            return None
        
        # Expand face boundary for better hair coverage
        dilated_face = morphology.dilation(face_mask, morphology.disk(5))
        face_boundary = dilated_face & ~face_mask
        
        # Get upper region of face for hair seeding
        y1 = max(0, int(bbox[1]))
        y2 = int(face_mask.shape[0] * 0.4)  # Consider only top 40% for hair
        face_boundary[:y1] = 0
        face_boundary[y2:] = 0
        
        # Initialize multiple seeds from face boundary
        seeds = face_boundary & candidates
        if not np.any(seeds):
            return None
        
        # Grow from multiple seed points
        result_mask = np.zeros_like(candidates)
        seed_points = np.where(seeds)
        
        # Take multiple seed points
        num_seeds = min(5, len(seed_points[0]))
        for i in range(num_seeds):
            idx = i * len(seed_points[0]) // num_seeds
            seed_y = seed_points[0][idx]
            seed_x = seed_points[1][idx]
            
            try:
                mask = morphology.flood_fill(
                    candidates.copy(),
                    (seed_y, seed_x),
                    1
                )
                result_mask |= mask
            except Exception as e:
                print(f"Error in region growing at seed {i}: {e}")
                continue
            
        return result_mask
        
    def validate_hair_regions(self, hair_mask, face_mask, bbox, confidence):
        """More flexible validation of hair regions"""
        if hair_mask is None:
            return None
            
        # Filter disconnected components first
        filtered_mask = self.filter_connected_components(hair_mask, face_mask)
        
        # Continue with existing validation logic using filtered_mask
        bbox = np.clip(bbox, 0, None)
        bbox = np.round(bbox).astype(int)
        
        hair_area = float(np.sum(filtered_mask))
        face_area = float(np.sum(face_mask))
        
        if face_area > 0:
            hair_face_ratio = hair_area / face_area
        else:
            print("Invalid face area")
            return None
            
        # More flexible ratio constraints
        if not (0.2 <= hair_face_ratio <= 3.0):  # Broader range
            print(f"Unusual hair-to-face ratio: {hair_face_ratio:.2f} but accepting")
            # Don't return None, continue processing
        
        # More flexible position validation
        hair_points = np.where(filtered_mask)
        if len(hair_points[0]) == 0:
            return None
            
        hair_center_y = np.mean(hair_points[0])
        face_top_y = np.min(np.where(face_mask)[0])
        
        # Allow some hair to be below face top
        if hair_center_y > face_top_y + (bbox[3] - bbox[1]) * 0.5:
            print("Hair center unusually low but accepting")
        
        # More lenient confidence threshold
        if confidence < 0.3:  # Reduced from 0.5
            print(f"Low confidence detection: {confidence:.2f}")
            return None
            
        return filtered_mask
        
    def filter_connected_components(self, mask, face_mask):
        """Filter hair mask components based on connectivity to face"""
        # Label connected components
        labeled_array, num_features = ndimage.label(mask)
        
        # Dilate face mask to check for connectivity
        dilated_face = morphology.dilation(face_mask, morphology.disk(3))
        
        # Keep track of valid components
        valid_mask = np.zeros_like(mask, dtype=bool)
        
        for i in range(1, num_features + 1):
            component = labeled_array == i
            component_area = np.sum(component)
            
            # Check if component touches dilated face mask
            touches_face = np.any(component & dilated_face)
            
            # Keep component if it's large enough and connected to face
            if touches_face and component_area > self.min_hair_area:
                valid_mask |= component
                
        return valid_mask