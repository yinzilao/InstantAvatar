import numpy as np
import cv2
import os
from collections import deque

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
            
            # Save individual detection crops with bounds checking
            for i, det in enumerate(list(self.history)[-3:]):
                bbox = det['bbox']
                # Ensure coordinates are within image bounds
                y1 = max(0, min(h, int(bbox[1])))
                y2 = max(0, min(h, int(bbox[3])))
                x1 = max(0, min(w, int(bbox[0])))
                x2 = max(0, min(w, int(bbox[2])))
                
                # Only save crop if dimensions are valid
                if y2 > y1 and x2 > x1 and (y2-y1) > 0 and (x2-x1) > 0:
                    crop = track_vis[y1:y2, x1:x2]
                    if crop.size > 0:  # Additional check that crop is not empty
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
        debug_data = {
            'history': list(self.history)
        }
        
        if len(self.history) < 2:
            debug_data['error'] = "Not enough detections for smoothing"
            return None, None, 0.0, debug_data
            
        # Get recent detections for smoothing
        recent_dets = list(self.history)[-self.smooth_window:]
        if not recent_dets:
            debug_data['error'] = "No recent detections"
            return None, None, 0.0, debug_data
            
        # Calculate confidence-weighted average for bbox
        total_weight = 0
        weighted_bbox = np.zeros(4)
        
        for det in recent_dets:
            weight = det['confidence'] * (0.8 ** (self.frame_idx - det['frame_idx']))
            weighted_bbox += det['bbox'] * weight
            total_weight += weight
            
        if total_weight == 0:
            return None, None, 0.0, {'error': "Total weight is zero"}
            
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
        
        debug_data.update({
            'smooth_bbox': smooth_bbox,
            'smooth_mask': smooth_mask,
            'smooth_conf': smooth_conf
        })
        
        return smooth_bbox, smooth_mask, smooth_conf, debug_data
    
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