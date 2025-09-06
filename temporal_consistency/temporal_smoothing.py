import cv2
import numpy as np
import torch
import sys
import os

# Add RAFT to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external_libs', 'RAFT'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external_libs', 'RAFT', 'core'))

from core.raft import RAFT
from core.utils import flow_viz
import argparse

class TemporalMaskRefiner:
    def __init__(self):
        self.prev_mask = None
        self.prev_frame = None
        
        # Initialize RAFT
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help='restore checkpoint')
        parser.add_argument('--path', help='dataset for evaluation')
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        args = parser.parse_args(args=[])
        
        self.model = torch.nn.DataParallel(RAFT(args))
        # Note: You'll need to download the pretrained RAFT model
        # self.model.load_state_dict(torch.load('models/raft-things.pth'))
        
        self.model = self.model.module
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()

    def refine(self, current_frame, current_mask):
        if self.prev_frame is not None:
            with torch.no_grad():
                # Convert frames to tensor
                prev_tensor = torch.from_numpy(self.prev_frame).permute(2, 0, 1).float()[None].to('cuda' if torch.cuda.is_available() else 'cpu')
                curr_tensor = torch.from_numpy(current_frame).permute(2, 0, 1).float()[None].to('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Compute optical flow
                flow_low, flow_up = self.model(prev_tensor, curr_tensor, iters=20, test_mode=True)
                
                # Warp previous mask using optical flow
                flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()
                h, w = flow_np.shape[:2]
                flow_map = np.array([[[x, y] for x in range(w)] for y in range(h)], dtype=np.float32) + flow_np
                warped_mask = cv2.remap(self.prev_mask, flow_map, None, cv2.INTER_LINEAR)
                
                # Combine with current mask
                current_mask = np.maximum(current_mask, warped_mask)

        self.prev_mask = current_mask.copy()
        self.prev_frame = current_frame.copy()
        return current_mask