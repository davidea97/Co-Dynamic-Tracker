from cotracker.predictor import CoTrackerOnlinePredictor


from sklearn.decomposition import PCA
import matplotlib.cm as cm
from sklearn.mixture import GaussianMixture

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
from tracker.utils.save_utils import make_video_from_frames, save_init_dynamic_estimation, save_refined_dynamic_visualization
from tracker.utils.track_utils import TrackerUtils

class BatchOnlineDynamicTracker():
    def __init__(self, intrinsics=None, grid_size=30, checkpoint="scaled_online.pth", search_window_len=8, track_window_len=1, verbose=True):
        
        self.intrinsics = intrinsics
        self.checkpoint = checkpoint
        self.window_len_search = search_window_len
        self.window_len_track = track_window_len
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
        self.verbose = verbose
        self.semantic_tracker = SemanticTracker(self.window_len_search)
        self.window_frames = []
        self.global_tracks = []
        self.global_visibilities = []
        self.global_refined_points_3d = []
        self.tracker_utils = TrackerUtils(self.intrinsics, self.window_len_search)
        self.dynamic_tracking_mode = False
        self.refined_previous_3d_points = {}
        self.refined_previous_2d_points = {}

    def _process_step(self, window_frames, is_first_step, grid_size, grid_query_frame, queries=None):
        video_chunk = (
            torch.tensor(
                # np.stack(window_frames[-self.model.step:]), device=self.device
                np.stack(window_frames[-len(window_frames):]), device=self.device
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
    
    def process_window(self, window_rgb_images, queries=None):
        """
        Process a window of images to track objects and generate 3D points.
        """
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

        return pred_2d_tracks, pred_visibility
    
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
        jump_threshold = 0.05 # Right one 0.05
        angles = []
        for i in range(1, len(track_array)-1):
            v1 = track_array[i] - track_array[i-1]
            v2 = track_array[i+1] - track_array[i]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angles.append(cos_angle)
        if len(angles) > 0:
            mean_angle = np.mean(angles)
        else:
            mean_angle = 0.0

        if len(diffs) < 2:
            dynamic = False
        elif max_jump > 2 * jump_threshold:
            dynamic = False
        else:
            # dynamic = spread > 0.03  # Right one 0.03
            dynamic = mean_angle > 0.85 and spread > 0.02

        return dynamic, spread, speed, mean_angle
        

    def get_dynamic_3D_points(self, pred_3d_tracks, window_counter=0):
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
        per_frame_raw_static = defaultdict(list)
        per_frame_raw_dynamic = defaultdict(list)

        # self.tracker_utils.visualize_tracks(track_3d, draw_spread=True, window_counter=window_counter)
        for n, track in track_3d.items():
            dynamic, spread, speed, mean_angle = self.is_dynamic(track)
            for point, t in zip(track, frame_map[n]):
                if dynamic:
                    per_frame_raw_dynamic[t].append((n, point, spread, speed, mean_angle))
                else:
                    per_frame_raw_static[t].append((n, point, spread, speed, mean_angle))
                
        return per_frame_raw_dynamic, per_frame_raw_static


    def get_3D_points(self, window_rgb_images, window_depth_images, window_camera_poses, pred_2d_tracks, pred_visibility, previous_3d_points=None):
        """
        Computes 3D tracks for 2D tracked keypoints using camera intrinsics and depth.
        If previous_3d_points is given, only computes 3D points for the last frame and appends them.
        """
        if previous_3d_points is None:
            pred_3d_tracks = {}
            start_frame = 0
            end_frame = len(window_rgb_images)
        else:
            pred_3d_tracks = dict(previous_3d_points)
            start_frame = len(window_rgb_images) - 1  # Only last frame
            end_frame = len(window_rgb_images)
            prev_ids = [tid for tid, _ in previous_3d_points[len(previous_3d_points) - 1]]

        tracks2d = pred_2d_tracks[0].cpu().numpy()
        visibility2d = pred_visibility[0].cpu().numpy()
        for t in range(start_frame, end_frame):
            pose = window_camera_poses[t]
            depth = window_depth_images[t] / 1000  # Convert mm to meters
            keypoints = []

            for n in range(tracks2d.shape[1]):
                # Skip if the track is not visible
                if not visibility2d[t, n]:
                    continue

                x, y = tracks2d[t, n]
                x, y = int(x), int(y)

                if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
                    z = depth[y, x]
                    if z == 0:
                        continue

                    X = (x - self.cx) * z / self.fx
                    Y = (y - self.cy) * z / self.fy
                    Z = z

                    cam_coords = np.array([X, Y, Z])
                    cam_coords_hom = np.append(cam_coords, 1.0)
                    world_coords = pose @ cam_coords_hom
                    
                    if previous_3d_points is None:
                        keypoints.append((n, world_coords[:3]))
                    else:
                        keypoints.append((prev_ids[n-1], world_coords[:3]))

            # Append to existing tracks or create new ones
            if previous_3d_points is None:
                pred_3d_tracks[t] = keypoints
            else:
                pred_3d_tracks[len(previous_3d_points)] = keypoints

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
        window_depth_images,
        window_camera_poses,
        pred_tracks,
        pred_visibility,
        per_frame_raw_dynamic,
        per_frame_raw_static,
        window_counter,
        dynamic_threshold=0.5,
        min_points_in_mask=3,
        output_dir="output_masks"
        ):

        refined_2d_points_with_ids_per_frame = {}
        refined_2d_points_with_ids_within_masks = {}

        tracks_2d = pred_tracks[0].cpu().numpy()
        visibility_2d = pred_visibility[0].cpu().numpy()
        
        for t, img in enumerate(window_rgb_images):
            if t == 0:
                refined_dynamic_ids, _, _, refined_inside_masks_ids = self.SAM_refining(
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

            refined_with_ids = []
            for n in refined_dynamic_ids:
                if not visibility_2d[t, n]:
                    continue
                x, y = tracks_2d[t, n]
                refined_with_ids.append((n, [x, y]))

            refined_2d_points_with_ids_per_frame[t] = refined_with_ids

            # Let's include the points that are inside the mask
            refined_within_masks = []
            for n in refined_inside_masks_ids:
                if not visibility_2d[t, n]:
                    continue
                x, y = tracks_2d[t, n]
                refined_within_masks.append((n, [x, y]))
            refined_2d_points_with_ids_within_masks[t] = refined_within_masks

        refined_3d_points = {}
        refined_3d_points_within_masks = {}
        refined_2d_points = {}
        refined_2d_points_no_ids = {}
        for t, refined_with_ids in refined_2d_points_with_ids_per_frame.items():
            points2D = [xy for (_, xy) in refined_with_ids]
            points3D = self.compute_3D_from_2D(
                points2D,
                window_depth_images[t],
                window_camera_poses[t]
            )
            
            refined_2d_points[t] = [
                (track_id, point2D) for (track_id, point2D) in zip([id for (id, _) in refined_with_ids], points2D)
            ]

            refined_2d_points_no_ids[t] = points2D

            refined_3d_points[t] = [
                (track_id, point3D) for (track_id, point3D) in zip([id for (id, _) in refined_with_ids], points3D)
            ]
        
        # Let's try to analyze all the points within the masks
        for t, refined_with_ids in refined_2d_points_with_ids_within_masks.items():
            points2D = [xy for (_, xy) in refined_with_ids]
            points3D = self.compute_3D_from_2D(
                points2D,
                window_depth_images[t],
                window_camera_poses[t]
            )

            refined_3d_points_within_masks[t] = [
                (track_id, point3D) for (track_id, point3D) in zip([id for (id, _) in refined_with_ids], points3D)
            ]

        # for t in range(len(window_rgb_images)-1):
        #     src_dict = {id: pt for id, pt in refined_3d_points[t]}
        #     tgt_dict = {id: pt for id, pt in refined_3d_points[t + 1]} 

        #     common_ids = set(src_dict.keys()) & set(tgt_dict.keys())

        #     src_points = np.array([src_dict[i] for i in common_ids])
        #     tgt_points = np.array([tgt_dict[i] for i in common_ids])

        #     if len(src_points) < 4 or len(tgt_points) < 4:
        #         continue

        #     _, fitness, rmse = self.tracker_utils.estimate_ransac_se3(src_points, tgt_points)
        #     # if fitness < 0.75:
        #     print(f"Frame {t}->{t+1} | Fitness: {fitness:.3f} ! RMSE: {rmse:.3f}")

        for t in range(len(window_rgb_images)-1):
            src_dict = {id: pt for id, pt in refined_3d_points_within_masks[t]}
            tgt_dict = {id: pt for id, pt in refined_3d_points_within_masks[t + 1]} 

            common_ids = set(src_dict.keys()) & set(tgt_dict.keys())

            src_points = np.array([src_dict[i] for i in common_ids])
            tgt_points = np.array([tgt_dict[i] for i in common_ids])

            if len(src_points) < 4 or len(tgt_points) < 4:
                continue

            T, fitness, rmse = self.tracker_utils.estimate_ransac_se3(src_points, tgt_points)
            # if fitness < 0.75:
            disp = tgt_points - src_points
            pca = PCA(n_components=2)
            pca.fit(disp)
            pca_value =  pca.explained_variance_ratio_[0]
            print(f"Frame {t}->{t+1} | Fitness: {fitness:.3f} | RMSE: {rmse:.3f} | PCA: {pca_value:.3f} | n. points: {len(common_ids)}")
            # print(T)

        return refined_2d_points, refined_3d_points, refined_2d_points_no_ids
    

    def SAM_refining(
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
        - For the generated mask:
            - If the percentage of dynamic points inside the mask exceeds `dynamic_threshold`:
                - Retrieve both dynamic and static points within the mask.
                - Compute mean and standard deviation of spread and speed for dynamic points.
                - Remove outliers among dynamic points (i.e., those with low speed/spread relative to the mean).
                - Reclassify static points as dynamic if their speed and spread are above half the mean.
            - If the number of points inside the mask is less than `min_points_in_mask`, the mask is discarded.
            - If the dynamic ratio is below the threshold, all points inside the mask are discarded.

        Returns:
            - refined_dynamic_ids: a set of validated dynamic point indices
            - refined_points2D: list of [x, y] refined dynamic points (for visualization or tracking)
        """

        dynamic_points2D = []
        dynamic_ids = []
        for (n, _, _, _, _) in per_frame_raw_dynamic.get(frame_idx, []):
            x, y = tracks_2d[frame_idx, n]
            dynamic_points2D.append([x, y])
            dynamic_ids.append(n)
        
        static_points2D = []
        for (n, _, _, _, _) in per_frame_raw_static.get(frame_idx, []):
            x, y = tracks_2d[frame_idx, n]
            static_points2D.append([x, y])

        # If no dynamic points are available, return empty results
        if len(dynamic_points2D) == 0:
            return set(), [], [], set()

        # Applica SAM2
        mask_arrays = mask_generator_fn(image=image, dynamic_points=dynamic_points2D, static_points=None, output_dir=output_dir, window_counter=window_counter, image_counter=frame_idx)

        refined_dynamic_ids = set()
        refined_dynamic_ids_within_mask = set()

        # At the moment we assume a single masks per frame but it is possible to have multiple blobs
        for mask in mask_arrays:
            h, w = mask.shape
            num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
            
            for label_id in range(1, num_labels):  # label 0 is background
                mask_ids_inside = []
                mask_blob = (labels == label_id)
                for n in range(tracks_2d.shape[1]):
                    x, y = tracks_2d[frame_idx, n]
                    x, y = int(x), int(y)
                    if 0 <= y < h and 0 <= x < w and mask_blob[y, x]:
                        mask_ids_inside.append(n)
                        # refined_dynamic_ids_within_mask.add(n)
                
                # N.B. it can be possible that the number of points inside the mask is more than the number of dynamic points used as prompts
                if len(mask_ids_inside) < min_points_in_mask:
                    continue  
                
                # Check the ratio of init dynamic points inside the mask
                dynamic_in_mask = [n for n in mask_ids_inside if n in dynamic_ids]
                static_in_mask = [n for n in mask_ids_inside if n not in dynamic_ids]
                ratio_dynamic = len(dynamic_in_mask) / len(mask_ids_inside)

                if ratio_dynamic > dynamic_threshold:
                    dyn_speed_by_id = {n: speed for (n, _, _, speed, _) in per_frame_raw_dynamic.get(frame_idx, []) if n in dynamic_in_mask}
                    dyn_spread_by_id = {n: spread for (n, _, spread, _, _) in per_frame_raw_dynamic.get(frame_idx, []) if n in dynamic_in_mask}
                    static_speed_by_id = {n: speed for (n, _, _, speed, _) in per_frame_raw_static.get(frame_idx, []) if n in static_in_mask}
                    static_spread_by_id = {n: spread for (n, _, spread, _, _) in per_frame_raw_static.get(frame_idx, []) if n in static_in_mask}

                    dyn_speeds = list(dyn_speed_by_id.values())
                    dyn_spreads = list(dyn_spread_by_id.values())

                    if len(dyn_speeds) == 0 or len(dyn_spreads) == 0:
                        continue

                    mean_speed = np.median(dyn_speeds)
                    mean_spread = np.median(dyn_spreads)

                    for n in mask_ids_inside:
                        refined_dynamic_ids_within_mask.add(n)
                        # Remove false positives (let's compare the speed and spread of the dynamic points in the mask with the mean, theoretically they should be higher than the mean if they really belong to a dynamic object
                        # otherwise if they are outlier they are probably moving slowly)
                        if n in dynamic_ids and n in dyn_speed_by_id and n in dyn_spread_by_id:
                            speed = dyn_speed_by_id[n]
                            spread = dyn_spread_by_id[n]
                            if speed > mean_speed/2 and spread > mean_spread/2:
                                refined_dynamic_ids.add(n)
                            else:
                                pass

                        # Relabel static points
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
        dynamic_points2D = []
        refined_with_ids = []
        for n in refined_dynamic_ids:
            x, y = tracks_2d[frame_idx, n]
            dynamic_points2D.append([x, y])
            refined_with_ids.append((n, [x, y]))

        # Take all the points within the mask
        # refined_inside_masks_ids = []
        # for n in mask_ids_inside:
        #     x, y = tracks_2d[frame_idx, n]
        #     refined_inside_masks_ids.append([n, [x, y]])

        return refined_dynamic_ids, dynamic_points2D, refined_with_ids, refined_dynamic_ids_within_mask

    def masks_and_queries_for_next_window(self, window_rgb_images, refined_points_per_frame, window_counter, last_frame_dynamic):

        final_masks = [None] * len(window_rgb_images)

        if len(refined_points_per_frame[0]) > 0:
            final_masks = self.semantic_tracker.window_mask_generator(
                rgb_images=window_rgb_images,
                tracks2d=refined_points_per_frame[0],
                window_counter=window_counter,
                output_dir="refined_masks_video"
            )

        last_frame_final_mask = final_masks[-1] if len(final_masks) > 0 else None
        if len(last_frame_dynamic) > 0 and last_frame_final_mask is not None:
            mask_h, mask_w = last_frame_final_mask.shape

            # Filter points that are inside the mask
            filtered_pts = []
            for x, y in last_frame_dynamic:
                xi, yi = int(round(x)), int(round(y))
                if 0 <= yi < mask_h and 0 <= xi < mask_w:  # Make sure index is within bounds
                    if last_frame_final_mask[yi, xi]:
                        filtered_pts.append([x, y])

            if len(filtered_pts) > 0:
                queries_np = np.array([[0, x, y] for x, y in filtered_pts], dtype=np.float32)
                queries_tensor = torch.from_numpy(queries_np)[None].to(self.device)
            else:
                queries_tensor = None  # no valid queries in mask
        else:
            queries_tensor = None

        return queries_tensor, final_masks
    

    def additional_queries_for_next_window(self, window_rgb_images, refined_points_per_frame, window_counter, last_frame_dynamic):
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

                num_extra_points = min(30, len(coords))
                if num_extra_points > 0:
                    chosen = coords[np.random.choice(len(coords), size=num_extra_points, replace=False)]
                    additional_points = [[0, float(x), float(y)] for x, y in chosen]

            additional_points = np.array(additional_points, dtype=np.float32).reshape(-1, 3)
            all_queries_np = np.concatenate([queries_np, np.array(additional_points, dtype=np.float32)], axis=0)
            queries_tensor = torch.from_numpy(all_queries_np)[None].to(self.device)
            queries_tensor = torch.from_numpy(queries_np)[None].to(self.device)
        else:
            queries_tensor = None  # fallback

        return queries_tensor, final_masks
    


    def window_dynamic_search_process(self, window_rgb_images, window_depth_images, window_camera_poses, window_counter=0, queries=None):
        
        # Process the window tracks
        pred_2d_tracks, pred_visibility = self.process_window(window_rgb_images, queries=queries)

        # Get 3D points with respect to the world reference frame from the 2D tracks (using depth and camera poses)
        pred_3d_tracks = self.get_3D_points(window_rgb_images, window_depth_images, window_camera_poses, pred_2d_tracks, pred_visibility)

        # Get dynamic and static points from the 3D tracks
        pred_3d_dynamic_tracks, pred_3d_static_tracks = self.get_dynamic_3D_points(pred_3d_tracks, window_counter=window_counter)
        
        # Raw dynamic and static points visualization
        refined_2d_points, refined_3d_points, refined_2d_points_no_ids = self.get_refined_dynamic_points(
            window_rgb_images,
            window_depth_images,
            window_camera_poses,
            pred_2d_tracks,
            pred_visibility,
            pred_3d_dynamic_tracks,
            pred_3d_static_tracks,
            window_counter=window_counter,
            dynamic_threshold=0.5,
            min_points_in_mask=4,
            output_dir="output_masks"
        )

        # Refine the 2D points to 3D points using depth and camera poses
        # refined_3d_points = {}
        # refined_2d_points = {}
        # for t, refined_with_ids in semantically_refined_2d_points_with_ids.items():
        #     points2D = [xy for (_, xy) in refined_with_ids]
        #     points3D = self.compute_3D_from_2D(
        #         points2D,
        #         window_depth_images[t],
        #         window_camera_poses[t]
        #     )
            
        #     refined_2d_points[t] = [
        #         (track_id, point2D) for (track_id, point2D) in zip([id for (id, _) in refined_with_ids], points2D)
        #     ]

        #     refined_3d_points[t] = [
        #         (track_id, point3D) for (track_id, point3D) in zip([id for (id, _) in refined_with_ids], points3D)
        #     ]

        # for t in range(len(window_rgb_images)-1):
        #     src_dict = {id: pt for id, pt in refined_3d_points[t]}
        #     tgt_dict = {id: pt for id, pt in refined_3d_points[t + 1]} 

        #     common_ids = set(src_dict.keys()) & set(tgt_dict.keys())

        #     src_points = np.array([src_dict[i] for i in common_ids])
        #     tgt_points = np.array([tgt_dict[i] for i in common_ids])

        #     if len(src_points) < 4 or len(tgt_points) < 4:
        #         continue

        #     _, fitness, rmse = self.tracker_utils.estimate_ransac_se3(src_points, tgt_points)
        #     # if fitness < 0.75:
        #     print(f"Frame {t}->{t+1} | Fitness: {fitness:.3f} ! RMSE: {rmse:.3f}")
        
        if self.verbose:
            save_init_dynamic_estimation(window_rgb_images, pred_2d_tracks, pred_3d_dynamic_tracks, pred_3d_static_tracks, window_counter=window_counter, window_len=self.window_len_search)
            save_refined_dynamic_visualization(window_rgb_images, pred_2d_tracks, refined_2d_points, output_dir="output_refined_visualization", window_counter=window_counter, window_len=self.window_len_search)

        # MEMORY BANK: let's keep only the last frame's refined points
        last_frame_idx = len(window_rgb_images) - 1
        last_frame_dynamic = refined_2d_points_no_ids[last_frame_idx]  # lista di [x, y]

        queries_tensor, final_masks = self.masks_and_queries_for_next_window(window_rgb_images, refined_2d_points_no_ids, window_counter, last_frame_dynamic)
        
        if len(refined_3d_points[0])>0:
            self.tracker_utils.align_3D_masks(window_rgb_images, final_masks, window_depth_images, window_camera_poses, window_counter, refined_3d_points)

        return pred_2d_tracks, pred_visibility, pred_3d_tracks, queries_tensor, refined_3d_points, final_masks, refined_2d_points
    

    def window_dynamic_tracking_process(self, window_rgb_images, window_depth_images, window_camera_poses, window_counter=0, queries=None, previous_3d_points=None, last_previous_2d_points=None):
        
        # Process the window tracks
        self._process_step(  
            window_rgb_images,
            is_first_step=True,
            grid_size=self.grid_size,
            grid_query_frame=self.grid_query_frame,
            queries=queries
        )

        pred_2d_tracks, pred_visibility = self._process_step(  
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
            pred_2d_tracks,
            previous_3d_points=previous_3d_points
        )

        per_frame_raw_dynamic, per_frame_raw_static = self.get_dynamic_3D_points(pred_3d_tracks)

        refined_2d_points_per_frame, refined_2d_points_with_ids_per_frame = self.get_refined_dynamic_points(
            window_rgb_images,
            pred_2d_tracks,
            per_frame_raw_dynamic,
            per_frame_raw_static,
            window_counter=window_counter,
            dynamic_threshold=0.6,
            min_points_in_mask=3,
            output_dir="output_masks"
        )

        save_init_dynamic_estimation(window_rgb_images, pred_2d_tracks, per_frame_raw_dynamic, per_frame_raw_static, window_counter=window_counter, window_len=self.window_len_search)
        save_refined_dynamic_visualization(window_rgb_images, pred_2d_tracks, per_frame_raw_dynamic, refined_2d_points_per_frame, output_dir="output_refined_visualization", window_counter=window_counter, window_len=self.window_len_search)

        # MEMORY BANK: let's keep only the last frame's refined points
        last_frame_idx = len(window_rgb_images) - 1
        last_frame_dynamic = refined_2d_points_per_frame[last_frame_idx]  # lista di [x, y]

        queries_tensor, final_masks = self.additional_queries_for_next_window(
            window_rgb_images,
            refined_2d_points_per_frame,
            window_counter,
            last_frame_dynamic
        )

        refined_3d_points = {}
        refined_2d_points = {}
        for t, refined_with_ids in refined_2d_points_with_ids_per_frame.items():
            points2D = [xy for (_, xy) in refined_with_ids]
            points3D = self.compute_3D_from_2D(
                points2D,
                window_depth_images[t],
                window_camera_poses[t]
            )

            refined_2d_points[t] = [
                (track_id, point2D) for (track_id, point2D) in zip([id for (id, _) in refined_with_ids], points2D)
            ]

            refined_3d_points[t] = [
                (track_id, point3D) for (track_id, point3D) in zip([id for (id, _) in refined_with_ids], points3D)
            ]
        
        if len(refined_3d_points[0])>0:
            self.tracker_utils.align_3D_masks(window_rgb_images, final_masks, window_depth_images, window_camera_poses, window_counter, refined_3d_points)

        return pred_2d_tracks, pred_visibility, pred_3d_tracks, queries_tensor, refined_3d_points, final_masks, refined_2d_points
    

    def online_search_and_dynamic_tracking(self, rgb_images, depth_images, camera_poses):
        
        self.dynamic_tracking_mode = False
        self.window_frames = []
        prev_queries = None
        window_counter = 0
        all_final_masks = []

        for i in tqdm(range(len(rgb_images))):
            
            idx_end = i
            # Phase 1: window-based dynamic search
            if not self.dynamic_tracking_mode:
                idx_start = i - self.window_len_search
                if i % self.window_len_search == 0 and i != 0:
                    print("Processing window from frame", idx_start, "to", idx_end-1)
                    # For the search phase, we do not use previous queries
                    pred_tracks, pred_visibility, _, prev_queries, refined_3d_points, final_masks, refined_2d_points = self.window_dynamic_search_process(
                        self.window_frames[idx_start:idx_end],
                        depth_images[idx_start:idx_end],
                        camera_poses[idx_start:idx_end],
                        window_counter=window_counter,
                        queries=None
                    )
                    window_counter += 1

                    self.global_tracks.append(pred_tracks[0])
                    self.global_visibilities.append(pred_visibility[0])
                    self.global_refined_points_3d.append(refined_3d_points)
                    all_final_masks.extend(final_masks)

                    # Check if any dynamic object is found
                    if any(len(v) > 0 for v in refined_3d_points.values()):
                        print(f"Dynamic object found at window {window_counter}. Switching to tracking mode.")
                        self.dynamic_tracking_mode = True
                        # Prepare queries for the next window
                        self.prev_query_tensor = prev_queries
                        # Store the last refined 3D points for the next window
                        temp_refined_points_3d = {k: v for k, v in refined_3d_points.items() if k != 0}
                        self.refined_previous_3d_points = {new_k: v for new_k, (_, v) in enumerate(temp_refined_points_3d.items())}

                        self.refined_previous_2d_points = refined_2d_points[self.window_len_search - 1] 

            # Phase 2: frame-by-frame dynamic tracking
            else:
                # The tracked points start from the frame i - window_len_track - 1
                idx_start = i - self.window_len_track - 1
                print("Processing window from frame", idx_start, "to", idx_end-1)
                pred_tracks, pred_visibility, _, prev_queries, refined_3d_points, final_masks, _ = self.window_dynamic_tracking_process(
                    self.window_frames[idx_start:idx_end],
                    depth_images[idx_start:idx_end],
                    camera_poses[idx_start:idx_end],
                    window_counter=window_counter,
                    queries=self.prev_query_tensor,
                    previous_3d_points=self.refined_previous_3d_points,
                    last_previous_2d_points=self.refined_previous_2d_points
                )

                window_counter += 1

                self.prev_query_tensor = prev_queries
                self.global_tracks.append(pred_tracks[0])
                self.global_visibilities.append(pred_visibility[0])
                self.global_refined_points_3d.append(refined_3d_points)
                all_final_masks.extend(final_masks)

                # Qui puoi anche interrompere se non trovi più l’oggetto dinamico per N frame
                # e ritornare a finestre disgiunte (Fase 1)
            self.window_frames.append(rgb_images[i])
        make_video_from_frames("output_refined_visualization", "refined_full_video.mp4")



    def batch_dynamic_tracking(self, rgb_images, depth_images, camera_poses):
        """
        Process a sequence of RGB images, depth images, and camera poses for dynamic tracking.
        This method processes the images in windows, tracking dynamic objects and refining their points.
        """
        
        prev_queries = None
        window_counter = 0
        all_final_masks = []
        for i in tqdm(range(0, len(rgb_images))):
            if i % self.window_len_search == 0 and i != 0:
                print("Processing window from frame", i - self.window_len_search, "to", i - 1)
                pred_tracks, pred_visibility, _, prev_queries, refined_3d_points, final_masks, refined_2d_points  = self.window_dynamic_search_process(
                    self.window_frames[i - self.window_len_search:i],
                    depth_images[i - self.window_len_search:i],
                    camera_poses[i - self.window_len_search:i],
                    window_counter=window_counter,
                    queries=None
                )

                window_counter += 1

                self.global_tracks.append(pred_tracks[0])  # Keep only second half
                self.global_visibilities.append(pred_visibility[0])
                self.global_refined_points_3d.append(refined_3d_points)
                all_final_masks.extend(final_masks)
            self.window_frames.append(rgb_images[i])
            

        # This handles the case where the last window is not prcocessed yet
        pred_tracks, pred_visibility, _, _, refined_3d_points, final_masks, refined_2d_points = self.window_dynamic_search_process(
            self.window_frames[-self.window_len_search:],
            depth_images[-self.window_len_search:],
            camera_poses[-self.window_len_search:],
            window_counter=window_counter,
            queries=None
        )
        all_final_masks.extend(final_masks)

        window_counter += 1

        self.global_tracks.append(pred_tracks[0])
        self.global_visibilities.append(pred_visibility[0])
        self.global_refined_points_3d.append(refined_3d_points)
        

        # self.tracker_utils.align_3D_masks(rgb_images, all_final_masks, depth_images, camera_poses)
    
        make_video_from_frames("output_refined_visualization", "refined_full_video.mp4")

