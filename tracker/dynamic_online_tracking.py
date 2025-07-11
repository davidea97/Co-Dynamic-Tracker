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
from tracker.semantic_tracker import SemanticTracker
from tracker.utils.save_utils import make_video_from_frames, save_dynamic_static_visualization, save_refined_dynamic_visualization

class OnlineDynamicTracker():
    def __init__(self, intrinsics=None, grid_size=30, checkpoint="scaled_online.pth", window_len=8):
        
        self.intrinsics = intrinsics
        self.checkpoint = checkpoint
        self.window_len = window_len
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
        self.semantic_tracker = SemanticTracker(self.window_len)
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


    def is_dynamic(self, track):
        """
        Check if a track is dynamic based on its spread and speed.
        Returns:
        - dynamic: True if the track is dynamic, False otherwise
        - spread: the spread of the track
        - speed: the speed of the track
        """
        track_array = np.array(track)
        center = np.median(track_array, axis=0)
        spread = np.median(np.linalg.norm(track_array - center, axis=1))
        speed = compute_velocity(track_array, dt=1.0)
        diffs = np.linalg.norm(np.diff(track_array, axis=0), axis=1)
        max_jump = np.max(diffs) if len(diffs) >= 2 else 0.0
        jump_threshold = 0.05

        if len(diffs) < 2:
            dynamic = False
        elif max_jump > 2 * jump_threshold:
            dynamic = False
        else:
            dynamic = spread > 0.03

        return dynamic, spread, speed
    
    def get_dynamic_3D_points(self, pred_3d_tracks):
        """
        Process the 3D tracks to identify dynamic and static points.
        Returns:
        - per_frame_dynamic: a dictionary mapping frame index to a list of dynamic points
        - per_frame_static: a dictionary mapping frame index to a list of static points
        """

        track_3d = defaultdict(list)
        frame_map = defaultdict(list)

        for t, keypoints in pred_3d_tracks.items():
            for n, point in keypoints:
                track_3d[n].append(point)
                frame_map[n].append(t)  

        # Output per frame
        per_frame_static = defaultdict(list)
        per_frame_dynamic = defaultdict(list)

        # Temporal consistency check
        for n, track in track_3d.items():
            dynamic, spread, speed = self.is_dynamic(track)

            for point, t in zip(track, frame_map[n]):
                if dynamic:
                    per_frame_dynamic[t].append((n, point, spread, speed, t))
                else:
                    per_frame_static[t].append((n, point, spread, speed))

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


    def get_refined_dynamic_points(self, 
        window_rgb_images, 
        pred_tracks,
        per_frame_raw_dynamic,
        per_frame_raw_static,
        window_counter,
        dynamic_threshold=0.5,
        min_points_in_mask=3,
        output_dir="output_masks"):
        """
        Refine dynamic points using SAM2 masks.
        Returns:
        - refined_points_per_frame: a dictionary mapping frame index to a list of refined dynamic points
        """

        refined_points_per_frame = {}

        tracks_2d = pred_tracks[0].cpu().numpy()

        for t, img in enumerate(window_rgb_images):
            if t == 0:
                refined_dynamic_ids, refined_points2D = self.refine_single_image_dynamic_points_from_mask(
                    image=img,
                    frame_idx=t,
                    tracks_2d=tracks_2d,
                    per_frame_raw_dynamic=per_frame_raw_dynamic,
                    per_frame_raw_static=per_frame_raw_static,
                    mask_generator_fn=self.semantic_tracker.mask_generator,
                    window_counter=window_counter,
                    dynamic_threshold=dynamic_threshold,
                    min_points_in_mask=min_points_in_mask,
                    output_dir=output_dir
                )
            else:
                refined_points2D = []
                for n in refined_dynamic_ids:
                    x, y = tracks_2d[t, n]
                    refined_points2D.append([x, y])

            refined_points_per_frame[t] = refined_points2D

        return refined_points_per_frame
    

    def refine_single_image_dynamic_points_from_mask(
        self,
        image,
        frame_idx,
        tracks_2d,
        per_frame_raw_dynamic,
        per_frame_raw_static,
        mask_generator_fn,
        window_counter,
        dynamic_threshold=0.5,
        min_points_in_mask=3,
        output_dir="output_masks"
    ):
        """
        At frame t=0 of a window:
        - Use raw dynamic points as prompts for SAM2 to generate masks.
        - For each generated mask:
            - If the percentage of dynamic points inside the mask exceeds `dynamic_threshold`:
                - Retrieve both dynamic and static points within the mask.
                - Compute mean and standard deviation of spread and speed for dynamic points.
                - Remove outliers among dynamic points (i.e., those with low speed/spread relative to the mean).
                - Reclassify static points as dynamic if their speed and spread fall within 1.5 standard deviations from the dynamic mean.
            - If the number of points inside the mask is less than `min_points_in_mask`, the mask is discarded.
            - If the dynamic ratio is below the threshold, all points inside the mask are discarded.

        Returns:
            - refined_dynamic_ids: a set of validated dynamic point indices
            - refined_points2D: list of [x, y] refined dynamic points (for visualization or tracking)
        """
        points2D = []
        dynamic_ids = []

        for (n, _, _, _, _) in per_frame_raw_dynamic.get(frame_idx, []):
            x, y = tracks_2d[frame_idx, n]
            points2D.append([x, y])
            dynamic_ids.append(n)

        if len(points2D) == 0:
            return set(), []

        # Applica SAM2
        mask_arrays = mask_generator_fn(image=image, tracks2d=points2D, output_dir=output_dir, window_counter=window_counter, image_counter=frame_idx)

        refined_dynamic_ids = set()

        for mask in mask_arrays:
            h, w = mask.shape
            mask_ids_inside = []

            for n in range(tracks_2d.shape[1]):
                x, y = tracks_2d[frame_idx, n]
                x, y = int(x), int(y)
                if 0 <= y < h and 0 <= x < w and mask[y, x]:
                    mask_ids_inside.append(n)

            if len(mask_ids_inside) < min_points_in_mask:
                continue  # scarta maschere troppo piccole

            dynamic_in_mask = [n for n in mask_ids_inside if n in dynamic_ids]
            static_in_mask = [n for n in mask_ids_inside if n not in dynamic_ids]
            ratio_dynamic = len(dynamic_in_mask) / len(mask_ids_inside)

            if ratio_dynamic >= dynamic_threshold:
                dyn_speed_by_id = {n: speed for (n, _, _, speed, _) in per_frame_raw_dynamic.get(frame_idx, []) if n in dynamic_in_mask}
                dyn_spread_by_id = {n: spread for (n, _, spread, _, _) in per_frame_raw_dynamic.get(frame_idx, []) if n in dynamic_in_mask}
                static_speed_by_id = {n: speed for (n, _, _, speed) in per_frame_raw_static.get(frame_idx, []) if n in static_in_mask}
                static_spread_by_id = {n: spread for (n, _, spread, _) in per_frame_raw_static.get(frame_idx, []) if n in static_in_mask}

                dyn_speeds = list(dyn_speed_by_id.values())
                dyn_spreads = list(dyn_spread_by_id.values())

                if len(dyn_speeds) == 0 or len(dyn_spreads) == 0:
                    continue

                mean_speed = np.median(dyn_speeds)
                mean_spread = np.median(dyn_spreads)

                for n in mask_ids_inside:
                    # Remove false positives
                    if n in dynamic_ids and n in dyn_speed_by_id and n in dyn_spread_by_id:
                        speed = dyn_speed_by_id[n]
                        spread = dyn_spread_by_id[n]
                        if speed > mean_speed/2 and spread > mean_spread/2:
                            refined_dynamic_ids.add(n)
                        else:
                            pass

                    # Refine static points
                    elif n in static_speed_by_id and n in static_spread_by_id:
                        speed = static_speed_by_id[n]
                        spread = static_spread_by_id[n]

                        # Convert static points to dynamic if their speed and spread are at least half of the mean
                        if speed > mean_speed/2 and spread > mean_spread/2:
                            refined_dynamic_ids.add(n)
            else:
                # All points in the mask are discarded
                pass 

        # Convert refined dynamic ids to 2D points
        refined_points2D = []
        for n in refined_dynamic_ids:
            x, y = tracks_2d[frame_idx, n]
            refined_points2D.append([x, y])

        return refined_dynamic_ids, refined_points2D

    def window_dynamic_tracking_process(self, window_rgb_images, window_depth_images, window_camera_poses, window_counter=0):
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

        per_frame_raw_dynamic, per_frame_raw_static = self.get_dynamic_3D_points(pred_3d_tracks)

        refined_points_per_frame = self.get_refined_dynamic_points(
            window_rgb_images,
            pred_tracks,
            per_frame_raw_dynamic,
            per_frame_raw_static,
            window_counter=window_counter,
            dynamic_threshold=0.4,
            min_points_in_mask=5,
            output_dir="output_masks"
        )

        save_dynamic_static_visualization(window_rgb_images, pred_tracks, per_frame_raw_dynamic, per_frame_raw_static, window_counter=window_counter, window_len=self.window_len)
        save_refined_dynamic_visualization(window_rgb_images, pred_tracks, per_frame_raw_dynamic, refined_points_per_frame, output_dir="output_refined_visualization", window_counter=window_counter, window_len=self.window_len)

        return pred_tracks, pred_visibility, pred_3d_tracks
    

    def full_online_dynamic_tracking(self, rgb_images, depth_images, camera_poses):
        
        self.global_tracks = []
        self.global_visibilities = []
        window_counter = 0
        for i in tqdm(range(0, len(rgb_images))):
            if i % self.window_len == 0 and i != 0:
                
                pred_tracks, pred_visibility, _ = self.window_dynamic_tracking_process(
                    self.window_frames[i - self.window_len:i],
                    depth_images[i - self.window_len:i],
                    camera_poses[i - self.window_len:i],
                    window_counter=window_counter
                )
                window_counter += 1

                self.global_tracks.append(pred_tracks[0])  # Keep only second half
                self.global_visibilities.append(pred_visibility[0])

            self.window_frames.append(rgb_images[i])

        # This handles the case where the last window is not prcocessed yet
        pred_tracks, pred_visibility, _ = self.window_dynamic_tracking_process(
            self.window_frames[-self.window_len:],
            depth_images[-self.window_len:],
            camera_poses[-self.window_len:],
            window_counter=window_counter
        )
        window_counter += 1

        self.global_tracks.append(pred_tracks[0])
        self.global_visibilities.append(pred_visibility[0])

        self.pred_tracks = torch.cat(self.global_tracks, dim=0)[None]  # (1, T, N, 2)
        self.pred_visibility = torch.cat(self.global_visibilities, dim=0)[None]
        make_video_from_frames("output_refined_visualization", "refined_full_video.mp4")