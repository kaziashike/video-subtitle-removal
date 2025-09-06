import cv2
import numpy as np
import torch
import sys
import os

# Add Real-ESRGAN to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external_libs', 'Real-ESRGAN'))

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    realesrgan_available = True
except ImportError:
    print("Warning: Real-ESRGAN modules not found. Using fallback upscaling method.")
    realesrgan_available = False

class SuperResolutionUpscaler:
    def __init__(self, model_path='experiments/pretrained_models/RealESRGAN_x4plus.pth'):
        if not realesrgan_available:
            return
            
        # Initialize Real-ESRGAN
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=True
        )

    def upscale_frame(self, frame):
        """
        Upscale a frame using Real-ESRGAN.
        
        Args:
            frame (np.ndarray): Input frame (H, W, 3)
            
        Returns:
            np.ndarray: Upscaled frame
        """
        if not realesrgan_available:
            # Fallback to simple interpolation
            height, width = frame.shape[:2]
            return cv2.resize(frame, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
            
        try:
            upscaled_frame, _ = self.upsampler.enhance(frame, outscale=2)
            return upscaled_frame
        except Exception as e:
            print(f"Error in Real-ESRGAN upscaling: {e}")
            # Fallback to simple interpolation
            height, width = frame.shape[:2]
            return cv2.resize(frame, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

# Initialize the upscaler
try:
    upscaler = SuperResolutionUpscaler()
    print("Super resolution upscaler initialized successfully")
except Exception as e:
    print(f"Failed to initialize super resolution upscaler: {e}")
    upscaler = None

def upscale_frame(frame):
    """
    Upscale a frame using Real-ESRGAN or fallback method.
    
    Args:
        frame (np.ndarray): Input frame (H, W, 3)
        
    Returns:
        np.ndarray: Upscaled frame
    """
    if upscaler is not None:
        return upscaler.upscale_frame(frame)
    else:
        # Fallback to simple interpolation
        height, width = frame.shape[:2]
        return cv2.resize(frame, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)