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
from tracker.utils.track_utils import TrackerUtils

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
        self.global_tracks = []
        self.global_visibilities = []
        self.global_refined_points_3d = []
        self.tracker_utils = TrackerUtils(self.intrinsics, self.window_len)

    def _process_step(self, window_frames, is_first_step, grid_size, grid_query_frame, queries=None):
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
            queries=queries,
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

    def compute_3D_from_2D(self, refined_points2D, depth_image, camera_pose):
        """
        Convert a list of 2D points to 3D world coordinates using depth and camera pose.

        Args:
            refined_points2D: list of [x, y] pixel coordinates
            depth_image: depth image in meters (H x W)
            camera_pose: 4x4 transformation matrix (camera-to-world)

        Returns:
            refined_points3D: list of (X, Y, Z) world coordinates
        """
        refined_points3D = []
        for (x, y) in refined_points2D:
            x, y = int(x), int(y)
            if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
                z = depth_image[y, x]/1000
                if z > 0:
                    X = (x - self.cx) * z / self.fx
                    Y = (y - self.cy) * z / self.fy
                    Z = z
                    cam_coords = np.array([X, Y, Z, 1.0])  # homogeneous
                    world_coords = camera_pose @ cam_coords
                    refined_points3D.append(world_coords[:3])
        return refined_points3D

    def get_refined_dynamic_points(self, 
        window_rgb_images, 
        pred_tracks,
        per_frame_raw_dynamic,
        per_frame_raw_static,
        window_counter,
        dynamic_threshold=0.5,
        min_points_in_mask=3,
        output_dir="output_masks"
        ):

        """
        Refine dynamic points using SAM2 masks.
        Returns:
        - refined_2d_points_per_frame: {frame_idx: list of [x, y]}
        - refined_2d_points_with_ids_per_frame: {frame_idx: list of (track_id, [x, y])}
        """

        refined_2d_points_per_frame = {}
        refined_2d_points_with_ids_per_frame = {}

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
            # else:
            refined_points2D = []
            refined_with_ids = []
            for n in refined_dynamic_ids:
                x, y = tracks_2d[t, n]
                refined_points2D.append([x, y])
                refined_with_ids.append((n, [x, y]))

            refined_2d_points_with_ids_per_frame[t] = refined_with_ids
            refined_2d_points_per_frame[t] = refined_points2D

        return refined_2d_points_per_frame, refined_2d_points_with_ids_per_frame
    

            
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

    
    

    def queries_for_next_window(self, window_rgb_images, refined_points_per_frame, window_counter, last_frame_dynamic):
        """
        Prepare queries for the next window based on the refined points from the last frame.
        Returns:
        - queries_tensor: a tensor of queries for the next window
        - final_masks: a list of masks for the last frame
        """

        final_masks = [None] * len(window_rgb_images)

        if len(refined_points_per_frame[0]) > 0:
            final_masks = self.semantic_tracker.window_mask_generator(
                rgb_images=window_rgb_images,
                tracks2d=refined_points_per_frame[0],
                window_counter=window_counter,
                output_dir="refined_masks_video"
            )

        if len(last_frame_dynamic) > 0:
            queries_np = np.array([[0, x, y] for x, y in last_frame_dynamic], dtype=np.float32)

            additional_points = []
            last_mask = final_masks[-1]
            if last_mask is not None:
                ys, xs = np.where(last_mask)  
                coords = np.stack([xs, ys], axis=1)

                num_extra_points = min(40, len(coords))
                if num_extra_points > 0:
                    chosen = coords[np.random.choice(len(coords), size=num_extra_points, replace=False)]
                    additional_points = [[0, float(x), float(y)] for x, y in chosen]

            additional_points = np.array(additional_points, dtype=np.float32).reshape(-1, 3)
            all_queries_np = np.concatenate([queries_np, np.array(additional_points, dtype=np.float32)], axis=0)
            queries_tensor = torch.from_numpy(all_queries_np)[None].to(self.device)
        else:
            queries_tensor = None  # fallback

        return queries_tensor, final_masks
    


    def window_dynamic_tracking_process(self, window_rgb_images, window_depth_images, window_camera_poses, window_counter=0, queries=None):
        self._process_step(  
            window_rgb_images,
            is_first_step=True,
            grid_size=self.grid_size,
            grid_query_frame=self.grid_query_frame,
            queries=queries
        )

        pred_2d_tracks, pred_visibility = self._process_step(  # Tracking
            window_rgb_images,
            is_first_step=False,
            grid_size=self.grid_size,
            grid_query_frame=self.grid_query_frame,
            queries=queries
        )

        pred_3d_tracks = self.get_3D_points(
            window_rgb_images,
            window_depth_images,
            window_camera_poses,
            pred_2d_tracks
        )

        per_frame_raw_dynamic, per_frame_raw_static = self.get_dynamic_3D_points(pred_3d_tracks)

        refined_2d_points_per_frame, refined_2d_points_with_ids_per_frame = self.get_refined_dynamic_points(
            window_rgb_images,
            pred_2d_tracks,
            per_frame_raw_dynamic,
            per_frame_raw_static,
            window_counter=window_counter,
            dynamic_threshold=0.4,
            min_points_in_mask=5,
            output_dir="output_masks"
        )

        save_dynamic_static_visualization(window_rgb_images, pred_2d_tracks, per_frame_raw_dynamic, per_frame_raw_static, window_counter=window_counter, window_len=self.window_len)
        save_refined_dynamic_visualization(window_rgb_images, pred_2d_tracks, per_frame_raw_dynamic, refined_2d_points_per_frame, output_dir="output_refined_visualization", window_counter=window_counter, window_len=self.window_len)

        # MEMORY BANK: let's keep only the last frame's refined points
        last_frame_idx = len(window_rgb_images) - 1
        last_frame_dynamic = refined_2d_points_per_frame[last_frame_idx]  # lista di [x, y]

        queries_tensor, final_masks = self.queries_for_next_window(
            window_rgb_images,
            refined_2d_points_per_frame,
            window_counter,
            last_frame_dynamic
        )

        refined_3d_points = {}
        for t, refined_with_ids in refined_2d_points_with_ids_per_frame.items():
            points2D = [xy for (_, xy) in refined_with_ids]
            points3D = self.compute_3D_from_2D(
                points2D,
                window_depth_images[t],
                window_camera_poses[t]
            )
            
            refined_3d_points[t] = [
                (track_id, point3D) for (track_id, point3D) in zip([id for (id, _) in refined_with_ids], points3D)
            ]
        
        # Track objects that have been refined
        # if len(refined_3d_points[0])>0:
        #     for t in range(len(refined_3d_points) - 1):
        #         src_points = np.array(refined_3d_points[t])     # punti del frame t
        #         tgt_points = np.array(refined_3d_points[t + 1]) # punti del frame t+1

        #         if len(src_points) < 6 or len(tgt_points) < 6:
        #             print(f"Frame {t}->{t+1}: Not enough points, skipping")
        #             continue

        #         T, fitness = estimate_icp_transform(src_points, tgt_points)
        #         print(f"Frame {t}->{t+1} | Fitness: {fitness:.3f}\n{T}")

        if len(refined_3d_points[0])>0:
            self.tracker_utils.align_3D_masks(window_rgb_images, final_masks, window_depth_images, window_camera_poses, window_counter, refined_3d_points)

        return pred_2d_tracks, pred_visibility, pred_3d_tracks, queries_tensor, refined_3d_points
    

    def full_online_dynamic_tracking(self, rgb_images, depth_images, camera_poses):
        """
        Process a sequence of RGB images, depth images, and camera poses for dynamic tracking.
        This method processes the images in windows, tracking dynamic objects and refining their points.
        """
        
        
        prev_queries = None
        window_counter = 0
        for i in tqdm(range(0, len(rgb_images))):
            if i % self.window_len == 0 and i != 0:
                
                pred_tracks, pred_visibility, _, prev_queries, refined_3d_points  = self.window_dynamic_tracking_process(
                    self.window_frames[i - self.window_len:i],
                    depth_images[i - self.window_len:i],
                    camera_poses[i - self.window_len:i],
                    window_counter=window_counter,
                    queries=prev_queries
                )
                window_counter += 1

                self.global_tracks.append(pred_tracks[0])  # Keep only second half
                self.global_visibilities.append(pred_visibility[0])
                self.global_refined_points_3d.append(refined_3d_points)

            self.window_frames.append(rgb_images[i])

        # This handles the case where the last window is not prcocessed yet
        pred_tracks, pred_visibility, _, _, refined_3d_points = self.window_dynamic_tracking_process(
            self.window_frames[-self.window_len:],
            depth_images[-self.window_len:],
            camera_poses[-self.window_len:],
            window_counter=window_counter,
            queries=prev_queries
        )
        

        window_counter += 1

        self.global_tracks.append(pred_tracks[0])
        self.global_visibilities.append(pred_visibility[0])
        self.global_refined_points_3d.append(refined_3d_points)

    
        make_video_from_frames("output_refined_visualization", "refined_full_video.mp4")