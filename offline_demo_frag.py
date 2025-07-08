
import numpy as np
import os
import time
import sys
sys.path.append("/home/allegro/davide_ws/co-tracker/sam2")
from sam2_tracker import Sam2Tracker
from dynamic_tracker import DynamicTracker
from offline_tracker import OfflineTracker



def main(experiments_path, grid_size, intrinsics, window_len=8):
    
    rgb_path = os.path.join(experiments_path, "rgb")
    depth_path = os.path.join(experiments_path, "depth")
    poses_path = os.path.join(experiments_path, "camera_poses")

    start_tracking_time = time.time()
    tracker = OfflineTracker(rgb_path, intrinsics, poses_path, grid_size=grid_size)
    # Track all the points in the video
    print("Starting online tracking...")
    tracker.online_tracking(window_len=window_len)
    print(f"Tracking time: {time.time() - start_tracking_time} seconds")

    # print("Initial length pred tracks:", len(tracker.pred_tracks[0]))
    pred_tracks = tracker.pred_tracks[0].cpu().numpy()  # Convert to numpy for easier handling
    pred_visibility = tracker.pred_visibility[0].cpu().numpy()  # Convert to numpy for easier handling

    rgb_images = tracker.extract_image_files(rgb_path) if rgb_path else []
    depth_images = tracker.extract_image_files(depth_path) if depth_path else []
    camera_poses = tracker.extract_poses(poses_path) if poses_path else []

    # print("Number of RGB images:", len(rgb_images))
    # print("Number of Depth images:", len(depth_images))

    dynamic_tracker = DynamicTracker(
        tracks=pred_tracks,
        rgb_images=rgb_images,
        depth_images=depth_images,
        camera_poses=camera_poses,
        intrinsics=intrinsics,
        # start_frame=0,  
        # end_frame=17 
        tracker_step_window=tracker.model.step,
        grid_size=grid_size
    )

    extract_3d_point_time = time.time()
    points_3d_world = dynamic_tracker.extract_3D_points(save_detections=False)
    print(f"Extracting time {time.time()-extract_3d_point_time} sec")

    compute_track_to_grid_index_time = time.time()
    track_to_grid_index = dynamic_tracker.compute_track_to_grid_index(grid_size=grid_size)
    print(f"Computing track to grid index time {time.time()-compute_track_to_grid_index_time} sec")
    # Extract dynamic points
    # static_points_clust_fin, dynamic_candidates, dynamic_points_clust_fin = dynamic_tracker.extract_dynamic_points_with_clustering_fin(points_3d_world)
    extract_dynamic_points_time = time.time()
    static_points, dynamic_points = dynamic_tracker.extract_dynamic_points(points_3d_world, track_to_grid_index)
    print("Extracting dynamic points time:", time.time() - extract_dynamic_points_time)
    # static_points_clust_fin, dynamic_points_clust_fin = dynamic_tracker.extract_dynamic_points_generalized(points_3d_world)
    dynamic_points_per_frame = dynamic_tracker.get_dynamic_points_2d(dynamic_points)
    
    # Draw dynamic points on the frames
    dynamic_tracker.draw_dynamic_points(static_points, dynamic_points, output_path="./detected_dynamic_keypoints")

    # dynamic_tracker.draw_points_by_spread(static_points, dynamic_points, output_path="./detected_dynamic_keypoints_spread")

    # Create sam2tracker  
    # sam2tracker = Sam2Tracker(rgb_images=rgb_path, tracks=pred_tracks, start_frame=0, end_frame=len(rgb_images), tracker_step_window=tracker.model.step)
    # sam2tracker.track(dynamic_points_per_frame)

# Main function
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Online Tracker Demo")
    parser.add_argument("--grid_size", type=int, default=15, help="Size of the grid for tracking")
    parser.add_argument("--window_len", type=int, default=8, help="Length of the tracking window")
    parser.add_argument("--experiments_path", type=str, default="/home/allegro/davide_ws/habitat-lab/FisherRF-active-mapping/experiments/GaussianSLAM/Eudora-results/gibson/", help="Path to the experiments folder")
    
    args = parser.parse_args()
    intrinsics = np.array([[128, 0, 128],
                            [0, 128, 128],
                            [0,   0,   1]])
    
    main(args.experiments_path, args.grid_size, intrinsics, args.window_len)


