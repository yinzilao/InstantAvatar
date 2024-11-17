from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import glob
import os
import argparse
import sys
from ultralytics import YOLO
from collections import deque
from skimage import feature, morphology

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.custom.utils.sam_utils import HeadPointEstimator, KEYPOINT_INDICES
from scripts.custom.utils.face_detector import FaceDetector
from scripts.custom.utils.head_segmentation import HairSegmenter
from scripts.custom.utils.temporal_tracker import HeadTracker
from scripts.custom.utils.detection_combiner import combine_detections, validate_combined_detection

CHECKPOINT = os.path.expanduser("./third_parties/segment-anything/ckpts/sam_vit_h_4b8939.pth")
MODEL = "vit_h"

class MaskGenerator:
    def __init__(self, predictor):
        self.predictor = predictor
        self.face_detector = FaceDetector()
        self.hair_segmenter = HairSegmenter()
        self.head_tracker = HeadTracker()
        self.frame_idx = 0
    
    def process_single_frame(self, img, pts, human_boxes):
        """Process a single frame to generate body and head masks"""
        self.predictor.set_image(img)
        print(f"\nProcessing frame {self.frame_idx}")
        
        # Create full keypoint array
        full_keypoints = np.zeros((max(KEYPOINT_INDICES.values()) + 1, 3))
        for i in range(len(pts)):
            full_keypoints[i] = pts[i]
        
        best_mask = None
        best_head_mask = None
        head_points = None
        head_bbox = None
        largest_area = 0
        
        for box in human_boxes:
            box_pts = get_points_in_box(full_keypoints, box)
            if len(box_pts) == 0:
                continue
            
            person_mask, area = self.get_person_mask(box_pts, box)
            if person_mask is None or area == 0:
                continue
            
            if area > largest_area:
                largest_area = area
                best_mask = person_mask
                
                head_result = create_head_mask(
                    keypoints=full_keypoints,
                    box=box,
                    predictor=self.predictor,
                    masked_image=img,
                    frame_idx=self.frame_idx,
                    face_detector=self.face_detector,
                    hair_segmenter=self.hair_segmenter,
                    head_tracker=self.head_tracker
                )
                
                if head_result is not None:
                    best_head_mask, head_points, head_bbox = head_result
        
        self.frame_idx += 1
        return best_mask, best_head_mask, head_points, head_bbox
    
    def get_person_mask(self, box_pts, box):
        """Generate full person mask using SAM"""
        try:
            # Create mask for valid points while preserving indices
            valid_mask = box_pts[..., 2] > 0.5
            valid_pts = box_pts[valid_mask]
            
            if len(valid_pts) == 0:
                return None, 0
                
            masks, scores, _ = self.predictor.predict(
                point_coords=valid_pts[:, :2],
                point_labels=np.ones(len(valid_pts)),
                box=box.reshape(2, 2),
                multimask_output=True
            )
            
            person_mask = masks.sum(axis=0) > 0
            return person_mask, np.sum(person_mask)
            
        except Exception as e:
            print(f"Error predicting person mask: {e}")
            return None, 0

def get_human_boxes(img):
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Run inference on image
    results = model(img)
    
    boxes = []
    # Filter for human detections (class 0 is person)
    for r in results[0].boxes.data:
        if r[5] == 0:  # Class index 0 is person
            x1, y1, x2, y2, conf, cls = r.tolist()
            # Only keep high confidence detections
            if conf > 0.5:
                # Convert to numpy array with correct shape
                boxes.append(np.array([x1, y1, x2, y2]))
                
    return np.array(boxes) if boxes else None

def get_points_in_box(points, box):
    x1, y1, x2, y2 = box
    mask = (points[:, 0] >= x1) & (points[:, 0] <= x2) & \
           (points[:, 1] >= y1) & (points[:, 1] <= y2)
    return points[mask]

def filter_mask_by_bbox(mask, bbox):
    """Keep only connected components that intersect with bbox"""
    print("\nMask Component Debug:")
    
    # Create bbox mask
    bbox_mask = np.zeros_like(mask)
    y1, x1 = max(0, int(bbox[1])), max(0, int(bbox[0]))
    y2, x2 = min(mask.shape[0], int(bbox[3])), min(mask.shape[1], int(bbox[2]))
    bbox_mask[y1:y2, x1:x2] = 1
    print(f"Created bbox mask with region: ({x1},{y1}) to ({x2},{y2})")
    
    # Find connected components
    from scipy import ndimage
    labeled_array, num_features = ndimage.label(mask)
    print(f"Found {num_features} connected components in mask")
    
    # Check which components intersect with bbox
    valid_components = []
    component_areas = []
    for i in range(1, num_features + 1):
        component = labeled_array == i
        intersection = component & bbox_mask
        if np.any(intersection):
            overlap_area = np.sum(intersection)
            component_area = np.sum(component)
            valid_components.append(i)
            component_areas.append(component_area)
            print(f"Component {i}:")
            print(f"  Total area: {component_area} pixels")
            print(f"  Overlap with bbox: {overlap_area} pixels ({overlap_area/component_area*100:.1f}%)")
        else:
            print(f"Component {i}: No intersection with bbox - filtered out")
    
    # Keep only valid components
    filtered_mask = np.zeros_like(mask)
    for i, comp_idx in enumerate(valid_components):
        comp_mask = labeled_array == comp_idx
        filtered_mask = filtered_mask | comp_mask
        print(f"Added component {comp_idx} to final mask (area: {component_areas[i]} pixels)")
    
    print(f"Final mask: kept {len(valid_components)} of {num_features} components")
    return filtered_mask

def create_head_mask(keypoints, box, predictor, masked_image, frame_idx, 
                    face_detector, hair_segmenter, head_tracker):
    """Create head mask using multiple detection strategies and temporal smoothing"""
    try:
        print(f"\nHead Detection Debug (Frame {frame_idx}):")
        
        # 1. Get keypoint-based head bbox
        estimator = HeadPointEstimator(keypoints, box)
        head_points, original_indices, background_points = estimator.get_head_points()
        keypoint_bbox = estimator.get_head_bbox()
        keypoint_conf = estimator.get_confidence() if keypoint_bbox is not None else 0.0
        print(f"Keypoint detection - Confidence: {keypoint_conf:.3f}")
        
        # 2. Get face detection bbox
        face_bbox, face_conf = face_detector.get_head_bbox(
            masked_image, 
            keypoint_bbox if keypoint_conf > 0.3 else None
        )
        print(f"Face detection - Confidence: {face_conf:.3f}")
        
        # 3. Get previous detection from tracker
        prev_bbox, prev_conf = head_tracker.get_last_detection()
        if prev_bbox is not None:
            print(f"Previous detection - Confidence: {prev_conf:.3f}")
        else:
            print("No previous detection found")
        
        # 4. Combine detections with confidence weighting
        final_bbox, confidence = combine_detections(
            keypoint_bbox, keypoint_conf,
            face_bbox, face_conf,
            prev_bbox, prev_conf
        )
        print(f"Combined detection - Confidence: {confidence:.3f}")
        
        # 5. Validate combined detection
        final_bbox, confidence = validate_combined_detection(
            final_bbox, confidence, masked_image.shape
        )
        print(f"Validated detection - Confidence: {confidence:.3f}")
        if final_bbox is None:
            print("No valid head detection")
            return None, None, None
            
        # 6. Get initial face mask from SAM
        masks, scores, _ = predictor.predict(
            point_coords=head_points,
            point_labels=np.ones(len(head_points)),
            box=final_bbox,
            multimask_output=True
        )
        
        if masks is None or len(masks) == 0:
            print("SAM prediction failed")
            return None, None, None
            
        face_mask = masks[np.argmax(scores)]
        
        # 7. Get hair mask with validation
        hair_mask = hair_segmenter.get_validated_hair_mask(
            masked_image, face_mask, final_bbox, confidence
        )
        
        # 8. Combine face and hair masks
        head_mask = face_mask.copy()
        if hair_mask is not None:
            head_mask |= hair_mask
            
        # 9. Update temporal tracker
        head_tracker.update(final_bbox, head_mask, confidence)
        
        # 10. Get smoothed prediction
        smooth_bbox, smooth_mask, smooth_conf = head_tracker.get_smooth_prediction()
        
        if smooth_mask is not None:
            print(f"Using smoothed prediction (conf: {smooth_conf:.3f})")
            return smooth_mask, (head_points, original_indices), smooth_bbox
            
        print(f"Using current prediction (conf: {confidence:.3f})")
        return head_mask, (head_points, original_indices), final_bbox
        
    except Exception as e:
        print(f"Error in create_head_mask: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def combine_face_hair_masks(face_mask, hair_mask):
    """Combine face and hair masks with validation"""
    if hair_mask is None:
        return face_mask
        
    # Ensure no conflicts between face and hair
    hair_mask = hair_mask & ~face_mask
    
    # Combine masks
    head_mask = face_mask | hair_mask
    
    return head_mask

def get_sam_mask(predictor, head_points, bbox):
    """Get face mask using SAM"""
    try:
        # Add background points
        background_points = generate_background_points(bbox)
        
        masks, scores, _ = predictor.predict(
            point_coords=np.vstack([head_points, background_points]),
            point_labels=np.concatenate([
                np.ones(len(head_points)),
                np.zeros(len(background_points))
            ]),
            box=bbox.reshape(2, 2),
            multimask_output=True
        )
        
        # Validate and select best mask
        valid_masks, valid_scores = validate_masks(masks, scores, bbox)
        if not valid_masks:
            return None
            
        return valid_masks[np.argmax(valid_scores)]
        
    except Exception as e:
        print(f"Error in get_sam_mask: {e}")
        return None

def validate_head_masks(masks, scores, box):
    """Validate masks based on size and position constraints"""
    valid_masks = []
    valid_scores = []
    box_height = box[3] - box[1]
    box_width = box[2] - box[0]
    
    print(f"\nDetailed Mask Validation Debug:")
    print(f"Box dimensions - Width: {box_width}, Height: {box_height}")
    print(f"Validation thresholds:")
    print(f"  Height: {0.1 * box_height:.1f} to {0.3 * box_height:.1f}")
    print(f"  Width: {0.1 * box_width:.1f} to {0.3 * box_width:.1f}")
    print(f"  Max relative Y: 0.4")
    
    for idx, mask in enumerate(masks):
        # Get mask properties
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            print(f"Mask {idx}: Empty mask")
            continue
            
        mask_height = np.max(y_coords) - np.min(y_coords)
        mask_width = np.max(x_coords) - np.min(x_coords)
        relative_y = (np.mean(y_coords) - box[1]) / box_height
        
        print(f"\nMask {idx} details:")
        print(f"  Height: {mask_height} pixels ({mask_height/box_height:.2%} of box)")
        print(f"  Width: {mask_width} pixels ({mask_width/box_width:.2%} of box)")
        print(f"  Relative Y position: {relative_y:.2%}")
        print(f"  Score: {scores[idx]:.3f}")
        
        # Validate size and position
        height_valid = 0.1 * box_height <= mask_height <= 0.3 * box_height
        width_valid = 0.1 * box_width <= mask_width <= 0.3 * box_width
        position_valid = relative_y <= 0.4
        
        print(f"  Validation results:")
        print(f"    Height valid: {height_valid} ({0.1:.1f} <= {mask_height/box_height:.1%} <= {0.3:.1f})")
        print(f"    Width valid: {width_valid} ({0.1:.1f} <= {mask_width/box_width:.1%} <= {0.3:.1f})")
        print(f"    Position valid: {position_valid} ({relative_y:.1%} <= 0.4)")
        
        if height_valid and width_valid and position_valid:
            valid_masks.append(mask)
            valid_scores.append(scores[idx])
            print("    ✓ Mask validated")
        else:
            print("    ✗ Mask rejected")
            
    return valid_masks, valid_scores

def save_masks_and_visualizations(img, fn, output_root, best_mask, best_head_mask=None, head_points=None, head_bbox=None):
    """Save all masks and visualizations"""
    if best_mask is None:
        return
        
    # Save full body mask
    cv2.imwrite(os.path.join(output_root, "masks_sam", os.path.basename(fn)), 
                best_mask.astype(np.uint8) * 255)
    
    # Create and save body-only mask (excluding head only)
    body_only_mask = best_mask.copy()
    if best_head_mask is not None:
        body_only_mask[best_head_mask] = 0  # Remove head region
    
    cv2.imwrite(os.path.join(output_root, "masks_body_only", os.path.basename(fn)), 
                body_only_mask.astype(np.uint8) * 255)
    
    # Save visualizations
    full_vis = img.copy()
    full_vis[~best_mask] = 0
    cv2.imwrite(os.path.join(output_root, "masks_sam_images", os.path.basename(fn)), 
                full_vis)
    
    body_vis = img.copy()
    body_vis[~body_only_mask] = 0
    cv2.imwrite(os.path.join(output_root, "masks_body_only_images", os.path.basename(fn)), 
                body_vis)
    
    # Debug visualization for head mask only
    if best_head_mask is not None:
        debug_vis = create_debug_visualization(img, best_head_mask, head_points)
        cv2.imwrite(os.path.join(output_root, "debug_head_masks", os.path.basename(fn)), 
                   debug_vis)
    
    # Save head bbox crop if available
    if head_bbox is not None:
        # Ensure coordinates are within image bounds
        x1 = max(0, int(head_bbox[0]))
        y1 = max(0, int(head_bbox[1]))
        x2 = min(img.shape[1], int(head_bbox[2]))
        y2 = min(img.shape[0], int(head_bbox[3]))
        
        # Crop and save the head bbox region
        head_bbox_crop = img[y1:y2, x1:x2].copy()
        head_bbox_path = os.path.join(output_root, "head_bbox_crops", os.path.basename(fn))
        os.makedirs(os.path.dirname(head_bbox_path), exist_ok=True)
        cv2.imwrite(head_bbox_path, head_bbox_crop)

def create_debug_visualization(img, head_mask=None, head_points_data=None):
    """Create debug visualization with head mask and points"""
    debug_vis = img.copy()
    
    # Draw head points if available
    if head_points_data is not None:
        head_points, point_indices = head_points_data
        for i, (point, idx) in enumerate(zip(head_points, point_indices)):
            # Red for original keypoints, blue for estimated points
            color = (0, 0, 255) if idx is not None else (255, 0, 0)
            cv2.circle(debug_vis, 
                      (int(point[0]), int(point[1])), 
                      3, color, -1)
    
    # Overlay head mask in blue
    if head_mask is not None:
        debug_vis[head_mask] = debug_vis[head_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    return debug_vis

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate body and head masks using SAM')
    
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Root directory containing the data')
    parser.add_argument('--image_folder', type=str, required=True,
                      help='Folder name containing the images (relative to data_dir)')
    
    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    args = parse_args()
    
    # Validate paths
    image_path = os.path.join(args.data_dir, args.image_folder)
    keypoints_path = os.path.join(args.data_dir, "keypoints.npy")
    
    if not os.path.exists(image_path):
        raise ValueError(f"Image folder not found: {image_path}")
    if not os.path.exists(keypoints_path):
        raise ValueError(f"Keypoints file not found: {keypoints_path}")
    
    # Initialize SAM
    sam = sam_model_registry[MODEL](checkpoint=CHECKPOINT)
    sam.to("cuda")
    predictor = SamPredictor(sam)
    mask_generator = MaskGenerator(predictor)
    
    # Load data
    img_lists = sorted(glob.glob(os.path.join(image_path, "*.png")))
    if not img_lists:
        raise ValueError(f"No PNG images found in {image_path}")
    
    keypoints = np.load(keypoints_path)
    if len(keypoints) != len(img_lists):
        raise ValueError(f"Number of keypoints ({len(keypoints)}) does not match number of images ({len(img_lists)})")
    
    # Create output directories
    subdirs = [
        "masks_sam",
        "masks_sam_images", 
        "masks_body_only",
        "masks_body_only_images",
        "debug_head_masks",
        "head_bbox_crops"
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(args.data_dir, subdir), exist_ok=True)
    
    # Process each frame
    print(f"Processing {len(img_lists)} frames...")
    for idx, (fn, pts) in enumerate(zip(img_lists, keypoints)):
        print(f"Processing frame {idx+1}/{len(img_lists)}: {os.path.basename(fn)}")
        
        img = cv2.imread(fn)
        if img is None:
            print(f"Error reading image: {fn}")
            continue
            
        # Detect humans
        human_boxes = get_human_boxes(img)
        if human_boxes is None:
            print(f"No humans detected in {fn}")
            continue
        
        # Generate masks
        best_mask, best_head_mask, head_points, head_bbox = mask_generator.process_single_frame(
            img, pts, human_boxes
        )
        
        # Save results
        save_masks_and_visualizations(
            img, fn, args.data_dir, best_mask, best_head_mask, head_points, head_bbox
        )
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
