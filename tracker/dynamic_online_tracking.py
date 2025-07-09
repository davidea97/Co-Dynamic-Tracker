from cotracker.predictor import CoTrackerOnlinePredictor

import imageio.v3 as iio
import torch
import numpy as np
from tqdm import tqdm
from cotracker.utils.visualizer import Visualizer
import time


class OnlineDynamicTracker():
    def __init__(self, intrinsics=None, grid_size=30, checkpoint="scaled_online.pth"):
        
        self.intrinsics = intrinsics
        self.checkpoint = checkpoint
        self.grid_query_frame = 0
        self.grid_size = grid_size

        if self.checkpoint is not None:
            self.model = CoTrackerOnlinePredictor(checkpoint=self.checkpoint)
        else:
            self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
        self.device = 'cuda'
        self.model = self.model.to(self.device)

        self.window_frames = []

    def _process_step(self, window_frames, is_first_step, grid_size, grid_query_frame):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-self.model.step * 2 :]), device=self.device
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return self.model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )
    


    def window_dynamic_tracking_process(self, window_rgb_images, depth_images, camera_poses, window_len=8):
        self._process_step(  
            window_rgb_images,
            is_first_step=True,
            grid_size=self.grid_size,
            grid_query_frame=self.grid_query_frame,
        )

        pred_tracks, pred_visibility = self._process_step(  # Tracking
            window_rgb_images,
            is_first_step=False,
            grid_size=self.grid_size,
            grid_query_frame=self.grid_query_frame,
        )

        return pred_tracks, pred_visibility
    

    def full_online_dynamic_tracking(self, rgb_images, depth_images, camera_poses, window_len=8):

        self.global_tracks = []
        self.global_visibilities = []

        for i in tqdm(range(0, len(rgb_images))):
            if i % window_len == 0 and i != 0:
                
                pred_tracks, pred_visibility = self.window_dynamic_tracking_process(
                    self.window_frames[i - window_len:i],
                    depth_images[i - window_len:i],
                    camera_poses[i - window_len:i],
                    window_len=window_len
                )

                self.global_tracks.append(pred_tracks[0])  # Keep only second half
                self.global_visibilities.append(pred_visibility[0])

            self.window_frames.append(rgb_images[i])

        # This handles the case where the last window is not prcocessed yet
        pred_tracks, pred_visibility = self.window_dynamic_tracking_process(
            self.window_frames[-window_len:],
            depth_images[-window_len:],
            camera_poses[-window_len:],
            window_len=window_len
        )
        
        self.global_tracks.append(pred_tracks[0])
        self.global_visibilities.append(pred_visibility[0])

        self.pred_tracks = torch.cat(self.global_tracks, dim=0)[None]  # (1, T, N, 2)
        self.pred_visibility = torch.cat(self.global_visibilities, dim=0)[None]
