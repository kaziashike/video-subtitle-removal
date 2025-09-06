import cv2
import numpy as np
import torch
import sys
import os

# Add LaMa to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external_libs', 'lama'))

# Try to import LaMa modules
try:
    from saicinpainting.evaluation.refinement import refine_predict
    from saicinpainting.evaluation.utils import load_yaml
    from saicinpainting.training.trainers import load_checkpoint
    lama_available = True
except ImportError:
    print("Warning: LaMa modules not found. Using fallback inpainting method.")
    lama_available = False

class LaMaInpainter:
    def __init__(self, checkpoint_path="./external_libs/lama/big-lama", config_path="./external_libs/lama/configs/prediction/default.yaml"):
        if not lama_available:
            return
            
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load configuration
            self.config = load_yaml(config_path)
            
            # Load model
            self.model = load_checkpoint(self.config, checkpoint_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Failed to initialize LaMa inpainter: {e}")
            self.model = None

    def inpaint_frame(self, frame, mask):
        if not lama_available or self.model is None:
            # Fallback to simple inpainting using OpenCV
            mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            
        # Convert to float and normalize
        img = frame.astype(np.float32) / 255.0
        mask_normalized = mask.astype(np.float32) / 255.0
        
        # Convert to tensors
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask_normalized).unsqueeze(0).to(self.device)
        
        # Invert mask (LaMa expects 1 for inpainting region)
        mask_tensor = 1 - mask_tensor
        
        with torch.no_grad():
            # Run LaMa inpainting
            result = self.model(img_tensor, mask_tensor)
            
        # Convert back to numpy
        result_img = result[0].permute(1, 2, 0).cpu().numpy()
        result_img = (np.clip(result_img, 0, 1) * 255).astype(np.uint8)
        
        return result_img

# Initialize the inpainter
try:
    inpainter = LaMaInpainter()
    print("LaMa inpainter initialized successfully")
except Exception as e:
    print(f"Failed to initialize LaMa inpainter: {e}")
    inpainter = None

def inpaint_frame(frame, mask):
    """
    Inpaint a frame using LaMa or fallback method.
    
    Args:
        frame (np.ndarray): Input frame (H, W, 3)
        mask (np.ndarray): Mask of region to inpaint (H, W)
        
    Returns:
        np.ndarray: Inpainted frame
    """
    if inpainter is not None:
        return inpainter.inpaint_frame(frame, mask)
    else:
        # Fallback to OpenCV inpainting
        return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)