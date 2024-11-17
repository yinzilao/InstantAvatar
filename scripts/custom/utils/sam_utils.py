import numpy as np
import cv2
import os
from collections import deque

# OpenPose BODY_25 format indices
KEYPOINT_INDICES = {
    'nose': 0,
    'neck': 1,
    'right_shoulder': 2,
    'right_elbow': 3,
    'right_wrist': 4,
    'left_shoulder': 5,
    'left_elbow': 6,
    'left_wrist': 7,
    'mid_hip': 8,
    'right_hip': 9,
    'right_knee': 10,
    'right_ankle': 11,
    'left_hip': 12,
    'left_knee': 13,
    'left_ankle': 14,
    'right_eye': 15,
    'left_eye': 16,
    'right_ear': 17,
    'left_ear': 18,
    'left_big_toe': 19,
    'left_small_toe': 20,
    'left_heel': 21,
    'right_big_toe': 22,
    'right_small_toe': 23,
    'right_heel': 24
}

# Input configurations
PADDING_CONFIG = {
    'face': {
        'top': 1.5,
        'bottom': 0.3,
        'left': 0.75,
        'right': 0.75
    },
    'keypoint': {
        'top': 1.2,
        'bottom': 0.2,
        'left': 0.6,
        'right': 0.6
    }
}

CONFIDENCE_WEIGHTS = {
    'keypoint': 1.2,
    'face': 0.8,
    'temporal': 1.0
}

VALIDATION_THRESHOLDS = {
    'min_confidence': 0.3,
    'min_iou': 0.1,
    'min_size': 20,
    'max_aspect_ratio': 2.0
}

# Head keypoints subset
HEAD_KEYPOINT_INDICES = {
    'nose': KEYPOINT_INDICES['nose'],
    'left_eye': KEYPOINT_INDICES['left_eye'],
    'right_eye': KEYPOINT_INDICES['right_eye'],
    'left_ear': KEYPOINT_INDICES['left_ear'],
    'right_ear': KEYPOINT_INDICES['right_ear']
}

class HeadPointEstimator:
    def __init__(self, keypoints, box):
        """Initialize with keypoints and bounding box"""
        self.original_keypoints = keypoints
        self.keypoints = np.zeros((max(KEYPOINT_INDICES.values()) + 1, 3))
        
        print("\nKeypoint Index Debug:")
        print(f"Original keypoints shape: {keypoints.shape}")
        print(f"Full keypoints shape: {self.keypoints.shape}")
        
        for i in range(len(keypoints)):
            self.keypoints[i] = keypoints[i]
            if i in HEAD_KEYPOINT_INDICES.values():
                print(f"Head keypoint {i}: position={keypoints[i,:2]}, confidence={keypoints[i,2]}")
        
        self.box = box
        self.available_points = {}
        self.head_points = []
        
    def get_available_keypoints(self):
        """Get keypoints with confidence > 0.5"""
        print("\nKeypoint Detection Debug:")
        print("Checking head keypoints:")
        for name, idx in HEAD_KEYPOINT_INDICES.items():
            print(f"Looking for {name} at index {idx}")
            if idx < len(self.keypoints):
                confidence = self.keypoints[idx, 2]
                position = self.keypoints[idx, :2]
                print(f"{name}: index={idx}, position={position}, confidence={confidence:.2f}")
                
                if confidence > 0.5:
                    self.available_points[name] = position
                    self.head_points.append((position, idx))
                    print(f"  ✓ Added {name} at position {position} (index {idx})")
                else:
                    print(f"  ✗ Rejected {name} due to low confidence")
            else:
                print(f"  ! Index {idx} out of bounds")
    
    def estimate_nose_from_eyes(self):
        """Strategy 2: Estimate nose position from eyes"""
        print("\nNose Estimation Debug:")
        if 'nose' not in self.available_points and \
           any(eye in self.available_points for eye in ['left_eye', 'right_eye']):
            eye_points = [p for n, p in self.available_points.items() if 'eye' in n]
            if eye_points:
                eye_center = np.mean(eye_points, axis=0)
                nose_offset = np.array([0, (self.box[3] - self.box[1]) * 0.03])
                estimated_point = eye_center + nose_offset
                self.head_points.append((estimated_point, None))
                print(f"  ✓ Estimated nose at {estimated_point}")
    
    def estimate_eyes_from_ears(self):
        """Strategy 3: Estimate eye positions from ears"""
        print("\nEye Estimation Debug:")
        if not any('eye' in n for n in self.available_points) and \
           any('ear' in n for n in self.available_points):
            ear_points = [p for n, p in self.available_points.items() if 'ear' in n]
            if ear_points:
                ear_center = np.mean(ear_points, axis=0)
                eye_offset = np.array([(self.box[2] - self.box[0]) * 0.05, 0])
                estimated_point = ear_center + eye_offset
                self.head_points.append((estimated_point, None))
                print(f"  ✓ Estimated eye at {estimated_point}")
    
    def estimate_from_box(self):
        """Strategy 5: Fallback to box-based estimation"""
        print("\nBox Estimation Debug:")
        if len(self.head_points) < 2:
            box_width = self.box[2] - self.box[0]
            box_height = self.box[3] - self.box[1]
            head_height = box_height * 0.2
            
            # Add points as tuples with None index
            self.head_points = [
                (np.array([self.box[0] + box_width*0.5, self.box[1] + head_height*0.5]), None),
                (np.array([self.box[0] + box_width*0.3, self.box[1] + head_height*0.3]), None),
                (np.array([self.box[0] + box_width*0.7, self.box[1] + head_height*0.3]), None),
                (np.array([self.box[0] + box_width*0.3, self.box[1] + head_height*0.7]), None),
                (np.array([self.box[0] + box_width*0.7, self.box[1] + head_height*0.7]), None)
            ]
            print(f"  ✓ Added box points: {self.head_points}")

    def get_head_points(self):
        """Run all strategies in sequence"""
        print("\nHead Point Estimation Debug:") 
        self.get_available_keypoints()
        self.estimate_nose_from_eyes()
        self.estimate_eyes_from_ears()
        self.estimate_from_box()
        
        # Collect non-head keypoints as background points
        background_points = []
        print("\nBackground Points Debug:")
        for i, kp in enumerate(self.keypoints):
            # Skip if no confidence or is a head keypoint
            if kp[2] <= 0.5 or i in HEAD_KEYPOINT_INDICES.values():
                continue
            # Add high-confidence non-head keypoints as background
            background_points.append(kp[:2])
            print(f"Added background point {i}: position={kp[:2]}, confidence={kp[2]:.2f}")
        
        # Separate foreground and background points
        original_points = []
        original_indices = []  # Track original indices
        estimated_points = []
        
        for point, idx in self.head_points:
            if idx is not None:
                original_points.append(point)
                original_indices.append(idx)  # Store index
            else:
                estimated_points.append(point)
        
        # Return head points and background points separately
        return (np.array(original_points + estimated_points), 
                original_indices,
                np.array(background_points))

    def get_head_bbox(self):
        """Smart head bbox estimation with adaptive padding"""
        print("\nHead BBox Estimation Debug:")
        
        def estimate_head_orientation():
            """Estimate head orientation from available keypoints"""
            orientation = {
                'facing': 'front',  # default
                'confidence': 0.0
            }
            
            # Check left-right keypoint pairs
            pairs = [
                ('left_eye', 'right_eye'),
                ('left_ear', 'right_ear')
            ]
            
            visible_pairs = []
            for left, right in pairs:
                left_idx = HEAD_KEYPOINT_INDICES[left]
                right_idx = HEAD_KEYPOINT_INDICES[right]
                
                left_visible = self.keypoints[left_idx, 2] > 0.5
                right_visible = self.keypoints[right_idx, 2] > 0.5
                
                if left_visible and not right_visible:
                    orientation['facing'] = 'left' # left of the image, right side of the head
                    orientation['confidence'] += 0.5
                elif right_visible and not left_visible:
                    orientation['facing'] = 'right' # right of the image, left side of the head
                    orientation['confidence'] += 0.5
                elif left_visible and right_visible:
                    orientation['facing'] = 'front'
                    orientation['confidence'] += 1.0
                    
                if left_visible or right_visible:
                    visible_pairs.append((left_idx, right_idx))
            
            print(f"Estimated orientation: {orientation}")
            return orientation, visible_pairs
        
        def get_adaptive_padding(orientation):
            """Get adaptive padding based on head orientation"""
            base_padding = {
                'top': 1.0,    # More space for hair
                'bottom': 0.0, # Less space below face
                'left': 0.5,   # Medium space for sides
                'right': 0.5
            }
            
            # Adjust padding based on orientation
            if orientation['facing'] == 'left':
                base_padding['left'] *= 0.3   # Less padding on visible side
                base_padding['right'] *= 1.5   # More padding for back of head
            elif orientation['facing'] == 'right':
                base_padding['right'] *= 0.3
                base_padding['left'] *= 1.5
                
            # Scale padding by orientation confidence
            confidence_scale = 0.5 + (orientation['confidence'] * 0.5)
            for key in base_padding:
                base_padding[key] *= confidence_scale
                
            return base_padding
        
        # Get head orientation
        orientation, visible_pairs = estimate_head_orientation()
        
        # Collect relevant points
        relevant_points = []
        
        # Add detected head keypoints
        for point, idx in self.head_points:
            if idx is not None:
                relevant_points.append(point)
                print(f"Added head point: {point}")
        
        # Add neck and shoulders if visible
        for keypoint_name in ['neck', 'left_shoulder', 'right_shoulder']:
            idx = KEYPOINT_INDICES[keypoint_name]
            if self.keypoints[idx, 2] > 0.5:
                point = self.keypoints[idx, :2]
                relevant_points.append(point)
                print(f"Added {keypoint_name} point: {point}")
        
        if len(relevant_points) < 2:
            print("  ✗ Not enough points for bbox estimation")
            return None
        
        # Calculate base bbox
        points = np.array(relevant_points)
        min_xy = np.min(points, axis=0)
        max_xy = np.max(points, axis=0)
        
        # Get adaptive padding
        padding = get_adaptive_padding(orientation)
        
        # Calculate dimensions
        width = max_xy[0] - min_xy[0]
        height = max_xy[1] - min_xy[1]
        
        # Apply adaptive padding
        bbox = np.array([
            min_xy[0] - width * padding['left'],
            min_xy[1] - height * padding['top'],
            max_xy[0] + width * padding['right'],
            max_xy[1] + height * padding['bottom']
        ])
        
        print(f"  ✓ Generated head bbox with adaptive padding: {bbox}")
        print(f"  ✓ Padding ratios: {padding}")
        
        return bbox

    def get_confidence(self):
        """Calculate overall confidence of head detection"""
        if not self.available_points:
            self.get_available_keypoints()
            
        if not self.available_points:
            return 0.0
            
        # Get individual confidences with validation
        confidences = []
        for name, idx in HEAD_KEYPOINT_INDICES.items():
            if idx < len(self.keypoints):
                conf = float(self.keypoints[idx, 2])
                if not np.isnan(conf) and conf > 0.5:
                    confidences.append(conf)
                    
        if not confidences:
            return 0.0
            
        # Calculate weighted confidence
        # Give more weight to eyes and nose
        weights = {
            'nose': 1.2,
            'left_eye': 1.0,
            'right_eye': 1.0,
            'left_ear': 0.8,
            'right_ear': 0.8
        }
        
        total_weight = sum(weights.values())
        weighted_conf = sum(conf * weights[name] 
                          for conf, name in zip(confidences, self.available_points.keys()))
        
        return weighted_conf / total_weight

class HeadTracker:
    def __init__(self, history_length=5):
        """Initialize temporal tracker"""
        self.history = deque(maxlen=history_length)
        self.frame_idx = 0
        self.smooth_window = 3
        
    def update(self, bbox, mask, confidence):
        """Update tracker with new detection"""
        if bbox is None or mask is None:
            self.frame_idx += 1
            return
            
        # Store detection
        detection = {
            'bbox': bbox.copy(),
            'mask': mask.copy(),
            'confidence': confidence,
            'frame_idx': self.frame_idx
        }
        self.history.append(detection)
        
        # Save debug visualizations
        debug_dir = os.path.join("debug_images", f"frame_{self.frame_idx:05d}", "temporal_tracker")
        os.makedirs(debug_dir, exist_ok=True)

        # Save tracking history visualization
        if len(self.history) > 1:
            # Create blank canvas for visualization
            h, w = mask.shape[:2]
            track_vis = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Different colors for temporal tracking
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
            
            # Show last 3 detections with different colors
            for i, det in enumerate(list(self.history)[-3:]):
                bbox = det['bbox']
                mask = det['mask']
                color = colors[i % len(colors)]
                
                # Draw mask
                track_vis[mask] = color
                
                # Draw bbox
                cv2.rectangle(track_vis, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            color, 2)
                
                # Add confidence text
                cv2.putText(track_vis, 
                           f"conf: {det['confidence']:.2f}", 
                           (int(bbox[0]), int(bbox[1]-5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1)
            
            cv2.imwrite(os.path.join(debug_dir, "tracking_history.png"), track_vis)
            
            # Save individual detection crops
            for i, det in enumerate(list(self.history)[-3:]):
                bbox = det['bbox']
                y1, y2 = max(0, int(bbox[1])), min(h, int(bbox[3]))
                x1, x2 = max(0, int(bbox[0])), min(w, int(bbox[2]))
                if y2 > y1 and x2 > x1:
                    crop = track_vis[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(debug_dir, f"detection_{i}_crop.png"), crop)
        
        self.frame_idx += 1
    
    def get_last_detection(self):
        """Get most recent valid detection"""
        if not self.history:
            return None, 0.0
            
        last_det = self.history[-1]
        return last_det['bbox'], last_det['confidence']
    
    def get_smooth_prediction(self):
        """Get temporally smoothed prediction"""
        if len(self.history) < 2:
            return None, None, 0.0
            
        # Get recent detections for smoothing
        recent_dets = list(self.history)[-self.smooth_window:]
        if not recent_dets:
            return None, None, 0.0
            
        # Calculate confidence-weighted average for bbox
        total_weight = 0
        weighted_bbox = np.zeros(4)
        
        for det in recent_dets:
            weight = det['confidence'] * (0.8 ** (self.frame_idx - det['frame_idx']))
            weighted_bbox += det['bbox'] * weight
            total_weight += weight
            
        if total_weight == 0:
            return None, None, 0.0
            
        smooth_bbox = weighted_bbox / total_weight
        
        # Combine masks with temporal decay
        h, w = recent_dets[0]['mask'].shape
        smooth_mask = np.zeros((h, w), dtype=bool)
        total_weight = 0
        
        for det in recent_dets:
            weight = det['confidence'] * (0.8 ** (self.frame_idx - det['frame_idx']))
            smooth_mask |= det['mask']  # Union of recent masks
            total_weight += weight
            
        # Calculate smoothed confidence
        smooth_conf = total_weight / len(recent_dets)
        
        # Save smoothing debug visualization
        debug_dir = os.path.join("debug_images", f"frame_{self.frame_idx:05d}", 
                                "temporal_tracker", "smoothing")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Visualize smoothed result
        smooth_vis = np.zeros((h, w, 3), dtype=np.uint8)
        smooth_vis[smooth_mask] = (0, 255, 0)  # Green for smoothed mask
        cv2.rectangle(smooth_vis, 
                     (int(smooth_bbox[0]), int(smooth_bbox[1])), 
                     (int(smooth_bbox[2]), int(smooth_bbox[3])), 
                     (255, 255, 0), 2)  # Yellow for smoothed bbox
        cv2.putText(smooth_vis, 
                   f"smooth conf: {smooth_conf:.2f}", 
                   (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        
        cv2.imwrite(os.path.join(debug_dir, "smoothed_prediction.png"), smooth_vis)
        
        return smooth_bbox, smooth_mask, smooth_conf
    
    def validate_temporal_consistency(self, bbox, mask, confidence):
        """Validate detection against temporal history"""
        if not self.history:
            return bbox, mask, confidence
            
        last_det = self.history[-1]
        
        # Calculate IoU with previous detection
        iou = self.calculate_bbox_iou(bbox, last_det['bbox'])
        
        # Calculate mask overlap
        if mask is not None and last_det['mask'] is not None:
            mask_overlap = np.sum(mask & last_det['mask']) / np.sum(mask | last_det['mask'])
        else:
            mask_overlap = 0
            
        # Adjust confidence based on temporal consistency
        temporal_conf = min(1.0, confidence * (0.5 + 0.5 * iou) * (0.5 + 0.5 * mask_overlap))
        
        # Save validation debug visualization
        debug_dir = os.path.join("debug_images", f"frame_{self.frame_idx:05d}", 
                                "temporal_tracker", "validation")
        os.makedirs(debug_dir, exist_ok=True)
        
        h, w = mask.shape
        valid_vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw previous detection in red
        valid_vis[last_det['mask']] = (0, 0, 255)
        cv2.rectangle(valid_vis, 
                     (int(last_det['bbox'][0]), int(last_det['bbox'][1])), 
                     (int(last_det['bbox'][2]), int(last_det['bbox'][3])), 
                     (0, 0, 255), 2)
        
        # Draw current detection in green
        valid_vis[mask] = (0, 255, 0)
        cv2.rectangle(valid_vis, 
                     (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), 
                     (0, 255, 0), 2)
        
        # Add metrics text
        cv2.putText(valid_vis, 
                   f"IoU: {iou:.2f}", 
                   (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        cv2.putText(valid_vis, 
                   f"Mask Overlap: {mask_overlap:.2f}", 
                   (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        cv2.putText(valid_vis, 
                   f"Temporal Conf: {temporal_conf:.2f}", 
                   (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        
        cv2.imwrite(os.path.join(debug_dir, "temporal_validation.png"), valid_vis)
        
        return bbox, mask, temporal_conf
    
    def calculate_bbox_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        # Convert to numpy arrays
        box1 = np.array(box1)
        box2 = np.array(box2)
        
        # Get coordinates of intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate area of intersection
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate areas of both boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate union area
        union_area = box1_area + box2_area - intersection_area
        
        # Return IoU
        if union_area == 0:
            return 0.0
        return intersection_area / union_area
