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
            """Estimate head orientation using OpenPose keypoints"""
            orientation = {
                'facing': 'front',  # default
                'direction': 'center',  # left/right/center of image
                'confidence': 0.0
            }
            
            # First determine if person is facing camera or away
            # Use nose and neck visibility
            nose_conf = self.keypoints[KEYPOINT_INDICES['nose'], 2]
            neck_conf = self.keypoints[KEYPOINT_INDICES['neck'], 2]
            
            # If nose is not visible but neck is, likely facing away
            if nose_conf < 0.5 and neck_conf > 0.5:
                orientation['facing'] = 'back'
                orientation['confidence'] += 0.8
            elif nose_conf > 0.5:
                orientation['facing'] = 'front'
                orientation['confidence'] += 0.8
            
            # Determine left/right direction in image space
            left_points = {
                'ear': self.keypoints[KEYPOINT_INDICES['left_ear'], 2],
                'eye': self.keypoints[KEYPOINT_INDICES['left_eye'], 2],
                'shoulder': self.keypoints[KEYPOINT_INDICES['left_shoulder'], 2]
            }
            
            right_points = {
                'ear': self.keypoints[KEYPOINT_INDICES['right_ear'], 2],
                'eye': self.keypoints[KEYPOINT_INDICES['right_eye'], 2],
                'shoulder': self.keypoints[KEYPOINT_INDICES['right_shoulder'], 2]
            }
            
            # Count visible points on each side
            left_visible = sum(1 for conf in left_points.values() if conf > 0.5)
            right_visible = sum(1 for conf in right_points.values() if conf > 0.5)
            
            if left_visible > right_visible:
                orientation['direction'] = 'left'
                orientation['confidence'] += 0.5
            elif right_visible > left_visible:
                orientation['direction'] = 'right'
                orientation['confidence'] += 0.5
            
            print(f"Orientation detection:")
            print(f"- Facing: {orientation['facing']}")
            print(f"- Direction: {orientation['direction']}")
            print(f"- Confidence: {orientation['confidence']}")
            
            return orientation
        
        def get_adaptive_padding(orientation):
            """Get adaptive padding based on head orientation"""
            print("\nAdaptive Padding Debug:")
            print(f"Input orientation: {orientation}")
            
            base_padding = {
                'top': 1.0,    # More space for hair
                'bottom': 0.0, # Less space below face
                'left': 0.5,   # Medium space for sides
                'right': 0.5
            }
            print(f"Initial base padding: {base_padding}")
            
            # Adjust padding based on orientation
            if orientation['facing'] == 'front':
                print("Case: Front facing")
                # For front-facing, pad more on the occluded side
                if orientation['direction'] == 'left':
                    print("Direction: Left - Adjusting padding")
                    base_padding['right'] *= 2.0  # More padding on right
                    base_padding['left'] *= 0.5   # Less padding on visible side
                    print(f"After adjustment: {base_padding}")
                elif orientation['direction'] == 'right':
                    print("Direction: Right - Adjusting padding")
                    base_padding['left'] *= 2.0   # More padding on left
                    base_padding['right'] *= 0.5  # Less padding on visible side
                    print(f"After adjustment: {base_padding}")
            else:  # facing back
                print("Case: Back facing")
                # For back-facing, pad more on the opposite side of visible direction
                if orientation['direction'] == 'left':
                    print("Direction: Left - Adjusting padding")
                    base_padding['right'] *= 2.0  # More padding where face extends
                    base_padding['left'] *= 0.5   # Less padding on back of head
                    print(f"After adjustment: {base_padding}")
                elif orientation['direction'] == 'right':
                    print("Direction: Right - Adjusting padding")
                    base_padding['left'] *= 2.0   # More padding where face extends
                    base_padding['right'] *= 0.5  # Less padding on back of head
                    print(f"After adjustment: {base_padding}")
            
            print(f"Final padding values: {base_padding}")
            return base_padding
        
        # Get head orientation
        orientation = estimate_head_orientation()
        
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
        
        # Get neck point info for return value
        neck_point = None
        is_detected = False
        if KEYPOINT_INDICES['neck'] < len(self.keypoints):
            neck_conf = self.keypoints[KEYPOINT_INDICES['neck'], 2]
            if neck_conf > 0.5:
                neck_point = self.keypoints[KEYPOINT_INDICES['neck'], :2]
                is_detected = True
        
        # If no neck point detected, estimate from shoulders
        if neck_point is None:
            left_shoulder_idx = KEYPOINT_INDICES['left_shoulder']
            right_shoulder_idx = KEYPOINT_INDICES['right_shoulder']
            if (left_shoulder_idx < len(self.keypoints) and 
                right_shoulder_idx < len(self.keypoints)):
                left_conf = self.keypoints[left_shoulder_idx, 2]
                right_conf = self.keypoints[right_shoulder_idx, 2]
                if left_conf > 0.5 and right_conf > 0.5:
                    left_shoulder = self.keypoints[left_shoulder_idx, :2]
                    right_shoulder = self.keypoints[right_shoulder_idx, :2]
                    neck_point = (left_shoulder + right_shoulder) / 2
                    is_detected = False
                    print(f"Estimated neck point: {neck_point}")
        
        if len(relevant_points) < 2:
            print("  ✗ Not enough points for bbox estimation")
            return None, (None, False)
        
        # Calculate base bbox
        points = np.array(relevant_points)
        min_xy = np.min(points, axis=0)
        max_xy = np.max(points, axis=0)
        
        # Get adaptive padding
        padding = get_adaptive_padding(orientation)
        
        # Calculate dimensions
        width = max_xy[0] - min_xy[0]
        height = max_xy[1] - min_xy[1]
        
        # Find nose and neck points
        nose_point = None
        neck_point = None
        
        # Get nose point
        if KEYPOINT_INDICES['nose'] < len(self.keypoints) and self.keypoints[KEYPOINT_INDICES['nose'], 2] > 0.5:
            nose_point = self.keypoints[KEYPOINT_INDICES['nose'], :2]
        
        # Get neck point
        if KEYPOINT_INDICES['neck'] < len(self.keypoints) and self.keypoints[KEYPOINT_INDICES['neck'], 2] > 0.5:
            neck_point = self.keypoints[KEYPOINT_INDICES['neck'], :2]
        
        # Calculate vertical bounds based on nose-neck distance
        if nose_point is not None and neck_point is not None:
            nose_neck_distance = np.linalg.norm(nose_point - neck_point)
            bbox = np.array([
                min_xy[0] - width * padding['left'],
                nose_point[1] - nose_neck_distance * 1.2,  # Top of head
                max_xy[0] + width * padding['right'],
                nose_point[1] + nose_neck_distance * 0.7   # Bottom of head
            ])
        else:
            # Fallback to original calculation if nose or neck not detected
            height = max_xy[1] - min_xy[1]
            bbox = np.array([
                min_xy[0] - width * padding['left'],
                min_xy[1] - height * padding['top'],
                max_xy[0] + width * padding['right'],
                max_xy[1] + height * padding['bottom']
            ])
        
        print(f"  ✓ Generated head bbox with adaptive padding: {bbox}")
        print(f"  ✓ Padding ratios: {padding}")
        
        # Return all values in expected order
        return bbox, (neck_point, is_detected)

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
