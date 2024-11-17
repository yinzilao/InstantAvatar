import numpy as np
import os
import cv2

def combine_detections(keypoint_bbox, keypoint_conf, face_bbox, face_conf, prev_bbox=None, prev_conf=0.0):
    """Combine multiple head detections with confidence weighting"""
    valid_boxes = []
    confidences = []
    weights = []
    
    # Add keypoint detection if valid
    if keypoint_bbox is not None and keypoint_conf > 0.3:
        valid_boxes.append(keypoint_bbox)
        confidences.append(keypoint_conf)
        weights.append(1.0)  # Base weight for keypoints
        
    # Add face detection if valid
    if face_bbox is not None and face_conf > 0.3:
        valid_boxes.append(face_bbox)
        confidences.append(face_conf)
        weights.append(1.2)  # Higher weight for face detection
        
    # Add previous detection for temporal consistency
    if prev_bbox is not None and prev_conf > 0.2:
        valid_boxes.append(prev_bbox)
        confidences.append(prev_conf)
        weights.append(0.8)  # Lower weight for temporal consistency
        
    if not valid_boxes:
        return None, 0.0
        
    # Calculate final weights
    confidences = np.array(confidences)
    weights = np.array(weights)
    final_weights = confidences * weights
    final_weights /= final_weights.sum()
    
    # Weighted average of boxes
    valid_boxes = np.array(valid_boxes)
    final_bbox = np.average(valid_boxes, weights=final_weights, axis=0)
    
    # Calculate combined confidence
    max_conf = np.max(confidences)
    avg_conf = np.average(confidences, weights=weights)
    final_conf = 0.7 * max_conf + 0.3 * avg_conf
    
    # Return debug data along with results
    debug_data = {
        'keypoint_detection': {
            'bbox': keypoint_bbox,
            'conf': keypoint_conf
        },
        'face_detection': {
            'bbox': face_bbox,
            'conf': face_conf
        },
        'previous_detection': {
            'bbox': prev_bbox,
            'conf': prev_conf
        },
        'final_detection': {
            'bbox': final_bbox,
            'conf': final_conf
        }
    }
    
    return final_bbox, final_conf, debug_data

def validate_combined_detection(final_bbox, confidence, image_shape):
    """Validate combined detection result with anatomical constraints"""
    if final_bbox is None or confidence < 0.3:
        return None, 0.0
        
    # Ensure bbox is within image bounds
    h, w = image_shape[:2]
    final_bbox = np.clip(final_bbox, [0, 0, 0, 0], [w, h, w, h])
    
    # Get bbox dimensions
    width = final_bbox[2] - final_bbox[0]
    height = final_bbox[3] - final_bbox[1]
    
    # Validate minimum size
    if width < 20 or height < 20:
        print("Head bbox too small")
        return None, 0.0
    
    # Validate aspect ratio (width/height)
    aspect_ratio = width / height
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        print(f"Invalid head aspect ratio: {aspect_ratio:.2f}")
        return None, 0.0
    
    # Validate position relative to image
    center_y = (final_bbox[1] + final_bbox[3]) / 2
    if center_y > h * 0.7:  # Head shouldn't be in bottom 30% of image
        print("Head position too low in image")
        return None, 0.0
        
    # Validate size relative to image
    area_ratio = (width * height) / (w * h)
    if area_ratio > 0.5:  # Head shouldn't occupy more than 50% of image
        print(f"Head area too large: {area_ratio:.2f}")
        return None, 0.0
        
    return final_bbox, confidence 