import cv2
import numpy as np

def feather_mask(mask, radius=5):
    """
    Apply edge feathering to a mask for smoother blending.
    
    Args:
        mask (np.ndarray): Binary mask (H, W)
        radius (int): Feathering radius
        
    Returns:
        np.ndarray: Feathered mask (H, W)
    """
    # Apply morphological opening to smooth mask edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply distance transform for smooth gradient
    dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    
    # Normalize distance transform to 0-255 range
    normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Invert so that subtitle region is 0 and surrounding area is 255
    feathered = 255 - normalized
    
    # Combine with original mask to preserve strong edges
    result = cv2.bitwise_and(feathered, mask)
    
    return result

def refine_mask(mask):
    """
    Apply additional refinement to mask.
    
    Args:
        mask (np.ndarray): Input mask (H, W)
        
    Returns:
        np.ndarray: Refined mask (H, W)
    """
    # Apply morphological closing to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply slight blur to smooth edges
    smoothed = cv2.GaussianBlur(closed, (3, 3), 0)
    
    return smoothed