"""
Main module for video subtitle removal application.

This module integrates all components of the video subtitle removal pipeline:
1. Text Detection using PaddleOCR + DBNet
2. Temporal Consistency using RAFT Optical Flow
3. Refinement with Edge Feathering
4. Inpainting using LaMa
5. Super-Resolution using Real-ESRGAN

The implementation follows a modular design where each processing step
is clearly separated and can be independently maintained or replaced.
"""

import cv2
import numpy as np
import os
import argparse
import yaml
import sys

# Add external libraries to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'external_libs'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'external_libs', 'RAFT'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'external_libs', 'RAFT', 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'external_libs', 'lama'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'external_libs', 'Real-ESRGAN'))

# Import our custom modules
from text_detection import Detect_Subtitles_with_PaddleOCR
from inpainting import Inpaint_with_LaMa
from refinement import Edge_Feathering
from temporal_consistency import Temporal_Smoothing
from super_resolution import Upscale


def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration parameters
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_video(input_path, output_path, config):
    """
    Main function to process video and remove subtitles.
    
    This function implements the complete pipeline:
    1. Text Detection
    2. Temporal Consistency
    3. Refinement (Edge Feathering)
    4. Inpainting
    5. Super-Resolution
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path to output video file
        config (dict): Configuration parameters
    """
    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize temporal consistency module
    temporal_refiner = Temporal_Smoothing.TemporalMaskRefiner()
    
    print(f"Processing video: {input_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}")
        
        # Step 1: Text Detection using PaddleOCR + DBNet
        # More accurate than EasyOCR, handles rotated text, shadows, and low contrast
        mask = Detect_Subtitles_with_PaddleOCR.detect_subtitle_mask(frame)
        
        # Step 2: Temporal Consistency using RAFT Optical Flow
        # Prevents flickering across frames
        mask = temporal_refiner.refine(frame, mask)
        
        # Step 3: Refinement with Edge Feathering
        # Creates smoother blending than vmake.ai
        feathered_mask = Edge_Feathering.feather_mask(mask)
        
        # Step 4: Inpainting using LaMa (Large Mask Inpainting)
        # SOTA for large holes, cleaner than Stable Diffusion for real-world scenes
        inpainted_frame = Inpaint_with_LaMa.inpaint_frame(frame, feathered_mask)
        
        # Step 5: Super-Resolution using Real-ESRGAN
        # Upscale back to 1080p with sharp details
        # Only apply if resolution is below target (e.g., not already 1080p)
        if height < 1080:
            upscaled_frame = Upscale.upscale_frame(inpainted_frame)
            # Resize to exact 1080p if needed
            upscaled_frame = cv2.resize(upscaled_frame, (1920, 1080))
            out.write(upscaled_frame)
        else:
            out.write(inpainted_frame)
    
    # Release resources
    cap.release()
    out.release()
    print(f"Processing complete. Output saved to: {output_path}")


def main():
    """
    Main entry point for the video subtitle removal application.
    
    Parses command line arguments and initiates the video processing pipeline.
    """
    parser = argparse.ArgumentParser(description='Remove subtitles from video')
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('output', help='Output video file path')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Process the video
    process_video(args.input, args.output, config)


if __name__ == "__main__":
    """
    Entry point when the script is run directly.
    
    Example usage:
    python main.py input_video.mp4 output_video.mp4
    """
    main()