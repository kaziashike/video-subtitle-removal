import cv2
import numpy as np
from paddleocr import PaddleOCR

# Initialize PaddleOCR
# Using default settings for better accuracy
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def detect_subtitle_mask(frame):
    """
    Detect subtitles in a frame using PaddleOCR and create a mask.
    
    Args:
        frame (np.ndarray): Input frame (H, W, 3)
        
    Returns:
        np.ndarray: Binary mask of detected text regions (H, W)
    """
    # Run PaddleOCR detection
    result = ocr.ocr(frame)
    
    # Create blank mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Draw bounding boxes on mask
    if result is not None and len(result) > 0 and result[0] is not None:
        # Extract polygons from the result
        ocr_result = result[0]
        if hasattr(ocr_result, 'json') and 'res' in ocr_result.json:
            # New PaddleOCR format
            polygons = ocr_result.json['res']['dt_polys']
        elif hasattr(ocr_result, 'dt_polys'):
            # Alternative format
            polygons = ocr_result.dt_polys
        else:
            # Old format
            polygons = [detection[0] for detection in result[0]]
            
        for polygon in polygons:
            # Convert to integer points
            points = np.array(polygon, dtype=np.int32)
            # Fill the polygon in the mask
            cv2.fillPoly(mask, [points], 255)
    
    return mask