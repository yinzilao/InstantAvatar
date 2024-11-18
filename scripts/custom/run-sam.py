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
        neck_info = None
        debug_data = {}
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
                print(f"\nHead result: {head_result}")
                if head_result is not None:
                    best_head_mask, head_points, head_bbox, neck_info, frame_debug = head_result
                    debug_data.update(frame_debug)
        
        self.frame_idx += 1
        return best_mask, best_head_mask, head_points, head_bbox, neck_info, debug_data
    
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
        final_bbox, neck_info = estimator.get_head_bbox()
        keypoint_conf = estimator.get_confidence() if final_bbox is not None else 0.0
        print(f"Keypoint detection - Confidence: {keypoint_conf:.3f}")
        
        # 2. Get face detection bbox with proper error handling
        try:
            face_bbox, face_conf, face_debug = face_detector.get_head_bbox(
                masked_image, 
                final_bbox if keypoint_conf > 0.3 else None
            )
        except ValueError:
            print("Face detection returned invalid format")
            face_bbox, face_conf, face_debug = None, 0.0, {'error': 'Invalid return format'}
            
        # 3. Get previous detection from tracker
        prev_bbox, prev_conf = head_tracker.get_last_detection()
        if prev_bbox is not None:
            print(f"Previous detection - Confidence: {prev_conf:.3f}")
        else:
            print("No previous detection found")
        
        # 4. Combine detections with confidence weighting
        final_bbox, confidence, combiner_debug = combine_detections(
            final_bbox, keypoint_conf,
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
            return None, None, None, None, {'error': 'No valid head detection'}
            
        # 6. Get initial face mask from SAM
        masks, scores, _ = predictor.predict(
            point_coords=head_points,
            point_labels=np.ones(len(head_points)),
            box=final_bbox,
            multimask_output=True
        )
        
        if masks is None or len(masks) == 0:
            print("SAM prediction failed")
            return None, None, None, None, {'error': 'SAM prediction failed'}
            
        face_mask = masks[np.argmax(scores)]
        
        # 7. Get hair mask with validation
        hair_result = hair_segmenter.get_validated_hair_mask(
            masked_image, face_mask, final_bbox, confidence
        )
        hair_mask = None
        hair_debug = {}
        if hair_result is not None:
            hair_mask, hair_debug = hair_result
        
        # 8. Combine face and hair masks
        head_mask = face_mask.copy()
        if hair_mask is not None:
            head_mask |= hair_mask
            
        # 9. Update temporal tracker
        head_tracker.update(final_bbox, head_mask, confidence)
        
        # 10. Get smoothed prediction
        smooth_bbox, smooth_mask, smooth_conf, tracker_debug = head_tracker.get_smooth_prediction()
        
        # Collect all debug data
        debug_data = {
            'face_detector': face_debug,
            'detection_combiner': combiner_debug,
            'hair_segmentation': hair_debug,
            'temporal_tracker': tracker_debug,
            'sam_predictions': {
                'masks': masks,
                'scores': scores,
                'selected_mask': face_mask
            },
            'final_results': {
                'mask': smooth_mask if smooth_mask is not None else head_mask,
                'bbox': smooth_bbox if smooth_bbox is not None else final_bbox,
                'confidence': smooth_conf if smooth_conf is not None else confidence
            }
        }
        
        if smooth_mask is not None:
            print(f"Using smoothed prediction (conf: {smooth_conf:.3f})")
            return smooth_mask, (head_points, original_indices), smooth_bbox, neck_info, debug_data
            
        print(f"Using current prediction (conf: {confidence:.3f})")
        return head_mask, (head_points, original_indices), final_bbox, neck_info, debug_data
        
    except Exception as e:
        print(f"Error in create_head_mask: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, {'error': str(e)}

def combine_face_hair_masks(face_mask, hair_mask):
    """Combine face and hair masks with validation"""
    if hair_mask is None:
        return face_mask
        
    # Ensure no conflicts between face and hair
    hair_mask = hair_mask & ~face_mask
    
    # Combine masks
    head_mask = face_mask | hair_mask
    
    return head_mask

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

def save_masks_and_visualizations(img, fn, output_root, best_mask, best_head_mask=None, head_points=None, head_bbox=None, neck_info=None, full_keypoints=None):  # Add full_keypoints parameter
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
    
    print(f"\nNeck info: {neck_info}")
    # Debug visualization for head mask only
    if best_head_mask is not None:
        debug_vis = create_debug_visualization(
            img, 
            best_head_mask, 
            head_points,
            neck_info,
            full_keypoints  # Pass full keypoints to visualization
        )
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

def create_debug_visualization(img, head_mask=None, head_points_data=None, neck_info=None, full_keypoints=None):
    """Create debug visualization with head mask, points, and neck point"""
    debug_vis = img.copy()
    
    # Draw all body keypoints if available
    if full_keypoints is not None:
        print("\nDrawing full body keypoints:")
        for name, idx in KEYPOINT_INDICES.items():
            if idx < len(full_keypoints):
                point = full_keypoints[idx]
                if point[2] > 0.5:  # Check confidence threshold
                    # Draw point
                    cv2.circle(debug_vis,
                             (int(point[0]), int(point[1])),
                             4, (0, 255, 0), -1)  # Green circle for body keypoints
                    
                    # Add label with keypoint name
                    label_pos = (int(point[0]), int(point[1] - 10))
                    cv2.putText(debug_vis,
                              name,
                              label_pos,
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.4,  # Smaller font size for readability
                              (0, 255, 0),  # Green text
                              1)
                    print(f"Drew {name} at ({point[0]:.1f}, {point[1]:.1f}) conf: {point[2]:.2f}")
    
    # Draw head points if available
    if head_points_data is not None:
        head_points, point_indices = head_points_data
        for i, (point, idx) in enumerate(zip(head_points, point_indices)):
            # Red for original keypoints, blue for estimated points
            color = (0, 0, 255) if idx is not None else (255, 0, 0)
            cv2.circle(debug_vis, 
                      (int(point[0]), int(point[1])), 
                      3, color, -1)
    
    # Draw neck point if available
    print(f"\nNeck info: {neck_info}")
    if neck_info is not None:
        neck_point, is_detected = neck_info
        print(f"Neck point: {neck_point}, Detected: {is_detected}")
        # Green for detected neck, yellow for estimated neck
        color = (0, 255, 0) if is_detected else (0, 255, 255)
        cv2.circle(debug_vis,
                  (int(neck_point[0]), int(neck_point[1])),
                  5, color, -1)  # Larger circle for neck point
        label = "Detected Neck" if is_detected else "Estimated Neck"
        cv2.putText(debug_vis,
                   label,
                   (int(neck_point[0]), int(neck_point[1] - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
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

def save_debug_images(img, video_name, frame_idx, **debug_data):
    """Save debug images and visualizations"""
    debug_root = os.path.join(video_name, "debug_images") 
    frame_debug_dir = os.path.join(debug_root, f"frame_{frame_idx:05d}")
    
    # Create debug directories
    subdirs = [
        "detection_combiner",
        "face_detector",
        "temporal_tracker",
        "hair_segmentation",
        "sam_predictions",
        "final_results"
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(frame_debug_dir, subdir), exist_ok=True)

    # Save original image
    cv2.imwrite(os.path.join(frame_debug_dir, "00_original.png"), img)

    # Save detection combiner debug
    if 'detection_combiner' in debug_data:
        det_debug = debug_data['detection_combiner']
        det_dir = os.path.join(frame_debug_dir, "detection_combiner")
        
        # Save all bboxes and their crops
        for name, det in det_debug.items():
            if det['bbox'] is not None:
                bbox = det['bbox']
                # Draw bbox on image
                vis = img.copy()
                cv2.rectangle(vis, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            (0, 255, 0), 2)
                cv2.putText(vis, f"{name}: {det['conf']:.2f}", 
                           (int(bbox[0]), int(bbox[1]-5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imwrite(os.path.join(det_dir, f"{name}_bbox.png"), vis)
                
                # Ensure valid crop coordinates
                h, w = img.shape[:2]
                y1 = max(0, min(h-1, int(bbox[1])))
                y2 = max(0, min(h, int(bbox[3])))
                x1 = max(0, min(w-1, int(bbox[0])))
                x2 = max(0, min(w, int(bbox[2])))
                
                # Only save crop if dimensions are valid
                if y2 > y1 and x2 > x1:
                    crop = img[y1:y2, x1:x2]
                    if crop.size > 0:  # Additional check that crop is not empty
                        cv2.imwrite(os.path.join(det_dir, f"{name}_crop.png"), crop)

    # Save face detector debug
    if 'face_detector' in debug_data:
        face_debug = debug_data['face_detector']
        face_dir = os.path.join(frame_debug_dir, "face_detector")
        
        # Save face detections visualization
        if face_debug['detections']:
            face_vis = img.copy()
            for det in face_debug['detections']:
                bbox = det.location_data.relative_bounding_box
                h, w = img.shape[:2]
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                
                cv2.rectangle(face_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(face_vis, f"conf: {det.score[0]:.2f}", 
                           (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Save face crop
                face_crop = img[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(face_dir, f"face_crop_{x1}_{y1}.png"), face_crop)
                
            cv2.imwrite(os.path.join(face_dir, "face_detections.png"), face_vis)

    # Save temporal tracker debug
    if 'temporal_tracker' in debug_data:
        track_debug = debug_data['temporal_tracker']
        track_dir = os.path.join(frame_debug_dir, "temporal_tracker")
        
        # Save tracking history visualization
        if track_debug['history']:
            track_vis = img.copy()
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
            
            for i, det in enumerate(track_debug['history'][-3:]):
                bbox = det['bbox']
                mask = det['mask']
                color = colors[i % len(colors)]
                
                # Draw mask and bbox
                track_vis[mask] = track_vis[mask] * 0.5 + np.array(color) * 0.5
                cv2.rectangle(track_vis, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            color, 2)
                
            cv2.imwrite(os.path.join(track_dir, "tracking_history.png"), track_vis)

    # Save hair segmentation debug
    if 'hair_segmentation' in debug_data:
        hair_debug = debug_data['hair_segmentation']
        hair_dir = os.path.join(frame_debug_dir, "hair_segmentation")
        
        # Save intermediate results
        for name, mask in hair_debug.items():
            if isinstance(mask, np.ndarray):
                # Save binary mask
                cv2.imwrite(os.path.join(hair_dir, f"{name}_mask.png"), 
                           mask.astype(np.uint8) * 255)
                # Save masked image
                masked = img.copy()
                masked[~mask] = 0
                cv2.imwrite(os.path.join(hair_dir, f"{name}_masked.png"), masked)

    # Save final results
    if 'final_results' in debug_data:
        final_dir = os.path.join(frame_debug_dir, "final_results")
        final = debug_data['final_results']
        
        if 'mask' in final:
            cv2.imwrite(os.path.join(final_dir, "final_mask.png"), 
                       final['mask'].astype(np.uint8) * 255)
            
            final_vis = img.copy()
            final_vis[~final['mask']] = 0
            cv2.imwrite(os.path.join(final_dir, "final_masked.png"), final_vis)

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
        print(f"\nProcessing frame {idx+1}/{len(img_lists)}: {os.path.basename(fn)}")
        
        # Load image
        img = cv2.imread(fn)
        if img is None:
            print(f"Failed to load image: {fn}")
            continue
            
        # Get keypoints and boxes for this frame
        human_boxes = get_human_boxes(img)
        if human_boxes is None:
            print(f"No humans detected in {fn}")
            continue
        
        # Generate masks
        best_mask, best_head_mask, head_points, head_bbox, neck_info, debug_data = mask_generator.process_single_frame(
            img, pts, human_boxes
        )
        
        print(f"\nNeck info: {neck_info}")
        # Save results and visualizations
        save_masks_and_visualizations(
            img, 
            os.path.basename(fn), 
            args.data_dir, 
            best_mask, 
            best_head_mask, 
            head_points, 
            head_bbox,
            neck_info  # Make sure to pass neck_info here
        )
        
        # Add human detection debug data
        debug_data['human_detection'] = {
            'boxes': human_boxes,
            'selected_box': human_boxes[0] if len(human_boxes) > 0 else None
        }
        
        # Save debug images
        save_debug_images(
            img=img,
            video_name=args.data_dir,
            frame_idx=idx,
            **debug_data
        )
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
