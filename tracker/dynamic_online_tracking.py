from cotracker.predictor import CoTrackerOnlinePredictor

import imageio.v3 as iio
import torch
import numpy as np
from tqdm import tqdm
from cotracker.utils.visualizer import Visualizer
import time
from collections import defaultdict
import os
import cv2
from tracker.utils.general_utils import compute_velocity

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
    
    def save_dynamic_static_visualization(self, window_rgb_images, pred_tracks, per_frame_dynamic, per_frame_static, output_dir="dynamic_static_visualization", window_counter=0, window_len=8):

        os.makedirs(output_dir, exist_ok=True)
        tracks_2d = pred_tracks[0].cpu().numpy()  # [T, N, 2]

        for t, img in enumerate(window_rgb_images):
            img_out = img.copy()
            frame_idx = t

            # Disegna i punti statici in verde
            for (n, _, _) in per_frame_static.get(frame_idx, []):
                x, y = tracks_2d[t, n]
                cv2.circle(img_out, (int(x), int(y)), 2, (0, 255, 0), -1)  # Verde

            # Disegna i punti dinamici in rosso
            for (n, _, _, _) in per_frame_dynamic.get(frame_idx, []):
                x, y = tracks_2d[t, n]
                cv2.circle(img_out, (int(x), int(y)), 2, (255, 0, 0), -1)  # Rosso

            filename = os.path.join(output_dir, f"frame_{window_counter*window_len + t:04d}.png")
            cv2.imwrite(filename, cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))


    def get_window_dynamic_3d_points(self, pred_3d_tracks):
        track_3d = defaultdict(list)
        frame_map = defaultdict(list)

        for t, keypoints in pred_3d_tracks.items():
            for n, point in keypoints:
                track_3d[n].append(point)
                frame_map[n].append(t)  

        # Output per frame
        per_frame_static = defaultdict(list)
        per_frame_dynamic = defaultdict(list)
        # raw_dynamic_by_window = defaultdict(list)

        # SINGULAR TEMPORAL CHECKING
        for n, track in track_3d.items():
            track_array = np.array(track)
            center = np.median(track_array, axis=0)
            spread = np.median(np.linalg.norm(track_array - center, axis=1))
            speed = compute_velocity(track_array, dt=1.0)
            diffs = np.linalg.norm(np.diff(track_array, axis=0), axis=1)
            max_jump = np.max(diffs) if len(diffs) >= 2 else 0.0
            jump_threshold = 0.05

            if len(diffs) < 2:
                is_dynamic = False
            elif max_jump > 2 * jump_threshold:
                is_dynamic = False
            else:
                is_dynamic = spread > 0.03
                # is_dynamic = np.median(speed) > 0.04 #and spread > 0.02

            # if is_dynamic:
            #     print(f"[Track {n}] Dynamic → spread={spread:.4f} | max_jump={max_jump:.4f} | len={len(track)} | mean speed={np.mean(speed):.4f} | median speed={np.median(speed):.4f}")
            # else:
            #     print(f"[Track {n}] Static → spread={spread:.4f} | max_jump={max_jump:.4f} | len={len(track)} | mean speed={np.mean(speed):.4f} | median speed={np.median(speed):.4f}")

            for point, t in zip(track, frame_map[n]):
                if is_dynamic:
                    per_frame_dynamic[t].append((n, point, spread, t))
                else:
                    per_frame_static[t].append((n, point, spread))

        return per_frame_dynamic, per_frame_static

    def get_3D_points(self, window_rgb_images, window_depth_images, window_camera_poses, window_tracks):
        """
        Pred tracks 3D are in the format 
        {t: [(track_id, (X, Y, Z)), ...], ...}
        where t is the frame index, track_id is the id of the track, and (X, Y, Z) are the 3D coordinates in the world frame.
        """
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

    def window_dynamic_tracking_process(self, window_rgb_images, window_depth_images, window_camera_poses, window_len=8, window_counter=0):
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

        per_frame_dynamic, per_frame_static = self.get_window_dynamic_3d_points(pred_3d_tracks)
        self.save_dynamic_static_visualization(window_rgb_images, pred_tracks, per_frame_dynamic, per_frame_static, window_counter=window_counter, window_len=window_len)
        return pred_tracks, pred_visibility, pred_3d_tracks
    

    def full_online_dynamic_tracking(self, rgb_images, depth_images, camera_poses, window_len=8):

        self.global_tracks = []
        self.global_visibilities = []
        window_counter = 0
        for i in tqdm(range(0, len(rgb_images))):
            if i % window_len == 0 and i != 0:
                
                pred_tracks, pred_visibility, _ = self.window_dynamic_tracking_process(
                    self.window_frames[i - window_len:i],
                    depth_images[i - window_len:i],
                    camera_poses[i - window_len:i],
                    window_len=window_len,
                    window_counter=window_counter
                )
                window_counter += 1

                self.global_tracks.append(pred_tracks[0])  # Keep only second half
                self.global_visibilities.append(pred_visibility[0])

            self.window_frames.append(rgb_images[i])

        # This handles the case where the last window is not prcocessed yet
        
        pred_tracks, pred_visibility, _ = self.window_dynamic_tracking_process(
            self.window_frames[-window_len:],
            depth_images[-window_len:],
            camera_poses[-window_len:],
            window_len=window_len,
            window_counter=window_counter
        )
        window_counter += 1

        self.global_tracks.append(pred_tracks[0])
        self.global_visibilities.append(pred_visibility[0])

        self.pred_tracks = torch.cat(self.global_tracks, dim=0)[None]  # (1, T, N, 2)
        self.pred_visibility = torch.cat(self.global_visibilities, dim=0)[None]
