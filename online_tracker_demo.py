
import numpy as np
import os
import time
import sys
sys.path.append("./sam2")
from sam2_tracker import Sam2Tracker
from tracker.dynamic_tracker import DynamicTracker
from tracker.dynamic_online_tracking import BatchOnlineDynamicTracker

from tracker.utils.general_utils import extract_image_files, extract_poses, str2bool


def main(experiments_path, grid_size, intrinsics, search_window_len=8, track_window_len=8, checkpoint="scaled_online.pth", verbose=False):
    
    rgb_path = os.path.join(experiments_path, "rgb")
    depth_path = os.path.join(experiments_path, "depth")
    poses_path = os.path.join(experiments_path, "camera_poses")

    rgb_images = extract_image_files(rgb_path) if rgb_path else []
    depth_images = extract_image_files(depth_path) if depth_path else []
    camera_poses = extract_poses(poses_path) if poses_path else []

    start_tracking_time = time.time()
    tracker = BatchOnlineDynamicTracker(intrinsics, grid_size=grid_size, checkpoint=checkpoint, search_window_len=search_window_len, track_window_len=track_window_len, verbose=verbose)

    print("Start tracking...")
    # tracker.batch_dynamic_tracking(rgb_images, depth_images, camera_poses)
    tracker.online_search_and_dynamic_tracking(rgb_images, depth_images, camera_poses)
    print(f"Tracking time: {time.time() - start_tracking_time} seconds")

# Main function
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Online Tracker Demo")
    parser.add_argument("--grid_size", type=int, default=15, help="Size of the grid for tracking")
    parser.add_argument("--search_window_len", type=int, default=8, help="Length of the search window")
    parser.add_argument("--track_window_len", type=int, default=1, help="Length of the track window")
    parser.add_argument("--experiments_path", type=str, default="/home/allegro/davide_ws/habitat-lab/FisherRF-active-mapping/experiments/GaussianSLAM/Eudora-results/gibson/", help="Path to the experiments folder")
    parser.add_argument("--checkpoint", type=str, default="/home/allegro/davide_ws/co-tracker/checkpoints/scaled_online.pth", help="Path to the checkpoint file")
    parser.add_argument("--verbose", type=str2bool, default=True, help="Enable verbose output")

    args = parser.parse_args()
    intrinsics = np.array([[128, 0, 128],
                            [0, 128, 128],
                            [0,   0,   1]])
    
    main(args.experiments_path, args.grid_size, intrinsics, args.search_window_len, args.track_window_len, args.checkpoint, args.verbose)


