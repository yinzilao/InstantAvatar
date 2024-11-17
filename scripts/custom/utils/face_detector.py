import cv2
import numpy as np
import mediapipe as mp
import os

DATA_ROOT = "data/custom/e1"

class FaceDetector:
    def __init__(self):
        """Initialize MediaPipe face detector"""
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full range detection
            min_detection_confidence=0.5
        )
    
    def get_head_bbox(self, image, keypoint_bbox=None):
        """Use SAM head bbox as primary reference"""
        try:
            results = self.face_detector.process(image)
            
            # Always start with SAM bbox
            if keypoint_bbox is None:
                print("Error: SAM head bbox is required")
                return None, 0.0, {'detections': []}
            
            if not results.detections:
                print("No face detected by MediaPipe")
                # Use SAM bbox with lower confidence
                return keypoint_bbox, 0.4, {'detections': []}
            
            # Get highest confidence face detection
            detection = max(results.detections, key=lambda x: x.score[0])
            
            # Convert MediaPipe bbox to absolute coordinates
            bbox = detection.location_data.relative_bounding_box
            h, w = image.shape[:2]
            face_bbox = [
                bbox.xmin * w,
                bbox.ymin * h,
                (bbox.xmin + bbox.width) * w,
                (bbox.ymin + bbox.height) * h
            ]
            
            # Calculate IoU with SAM bbox
            iou = self.calculate_bbox_iou(face_bbox, keypoint_bbox)
            print(f"Face-SAM IoU: {iou:.3f}")
            
            # Always return SAM bbox, but adjust confidence based on face detection
            if iou < 0.1:
                # Low IoU - use SAM bbox with lower confidence
                confidence = 0.4
            else:
                # High IoU - use face detection confidence
                confidence = detection.score[0]
            
            # Return debug data along with results
            debug_data = {
                'detections': results.detections if results.detections else [],
                'sam_bbox': keypoint_bbox,
                'face_bbox': face_bbox,
                'confidence': confidence,
                'iou': iou
            }
            
            return keypoint_bbox, confidence, {'detections': results.detections, 'debug': debug_data}
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            return None, 0.0, {'error': str(e)}
    
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
    
    def __del__(self):
        """Clean up MediaPipe resources"""
        self.face_detector.close() 