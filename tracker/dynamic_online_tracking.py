from cotracker.predictor import CoTrackerOnlinePredictor

import imageio.v3 as iio
import torch
import numpy as np
from tqdm import tqdm
from cotracker.utils.visualizer import Visualizer
import time
from collections import defaultdict


class OnlineDynamicTracker():
    def __init__(self, intrinsics=None, grid_size=30, checkpoint="scaled_online.pth"):
        
        self.intrinsics = intrinsics
        self.checkpoint = checkpoint
        self.grid_query_frame = 0
        self.grid_size = grid_size
        self.fx = intrinsics[0, 0] if intrinsics is not None else 1.0
        self.fy = intrinsics[1, 1] if intrinsics is not None else 1.0
        self.cx = intrinsics[0, 2] if intrinsics is not None else 1.0
        self.cy = intrinsics[1, 2] if intrinsics is not None else 1.0

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
    
    def get_3D_points(self, window_rgb_images, window_depth_images, window_camera_poses, window_tracks):
        pred_3d_tracks = {}
        tracks2d = window_tracks[0].cpu().numpy()
        
        # Iterate over the window of RGB images and extract 3D points
        for t in range(len(window_rgb_images)):
            
            pose = window_camera_poses[t]
            depth = window_depth_images[t]/1000

            all_depths = []
            keypoints = []

            # Iterate over all tracks in the current frame
            for n in range(tracks2d.shape[1]):
                x, y = tracks2d[t, n]
                x, y = int(x), int(y)
                if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
                    z = depth[y, x]
                    if z > 0:
                        all_depths.append(z)

            for n in range(tracks2d.shape[1]):
                x, y = tracks2d[t, n]
                x, y = int(x), int(y)

                if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
                    z = depth[y, x]   ## meters
                    if z == 0:
                        continue

                    X = (x - self.cx) * z / self.fx
                    Y = (y - self.cy) * z / self.fy
                    Z = z 
                    
                    cam_coords = np.array([X, Y, Z])
                    cam_coords_hom = np.append(cam_coords, 1.0) 
                    world_coords = pose @ cam_coords_hom   # 4D world point
                    keypoints.append((n, world_coords[:3])) 
            
            pred_3d_tracks[t] = keypoints

        return pred_3d_tracks

    def window_dynamic_tracking_process(self, window_rgb_images, window_depth_images, window_camera_poses, window_len=8):
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

        pred_3d_tracks = self.get_3D_points(
            window_rgb_images,
            window_depth_images,
            window_camera_poses,
            pred_tracks
        )

        return pred_tracks, pred_visibility, pred_3d_tracks
    
    def full_online_dynamic_tracking(self, rgb_images, depth_images, camera_poses, window_len=8):

        self.global_tracks = []
        self.global_visibilities = []

        for i in tqdm(range(0, len(rgb_images))):
            if i % window_len == 0 and i != 0:
                
                pred_tracks, pred_visibility, _ = self.window_dynamic_tracking_process(
                    self.window_frames[i - window_len:i],
                    depth_images[i - window_len:i],
                    camera_poses[i - window_len:i],
                    window_len=window_len
                )

                self.global_tracks.append(pred_tracks[0])  # Keep only second half
                self.global_visibilities.append(pred_visibility[0])

            self.window_frames.append(rgb_images[i])

        # This handles the case where the last window is not prcocessed yet
        pred_tracks, pred_visibility, _ = self.window_dynamic_tracking_process(
            self.window_frames[-window_len:],
            depth_images[-window_len:],
            camera_poses[-window_len:],
            window_len=window_len
        )

        self.global_tracks.append(pred_tracks[0])
        self.global_visibilities.append(pred_visibility[0])

        self.pred_tracks = torch.cat(self.global_tracks, dim=0)[None]  # (1, T, N, 2)
        self.pred_visibility = torch.cat(self.global_visibilities, dim=0)[None]
