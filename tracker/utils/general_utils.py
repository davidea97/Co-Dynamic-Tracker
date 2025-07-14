import natsort
import glob
import imageio.v3 as iio
import os
import numpy as np
import cv2
from pathlib import Path

def extract_image_files(folder):
        files_path = natsort.natsorted(glob.glob(os.path.join(folder, "*.png")))
        files = [iio.imread(file) for file in files_path]
        return files
    
def extract_poses(folder):
    files_path = natsort.natsorted(glob.glob(os.path.join(folder, "*.npy")))
    files = [np.load(file) for file in files_path]
    return files

def compute_velocity(track_array, dt=1.0):
    diffs = np.diff(track_array, axis=0)
    speeds = np.linalg.norm(diffs, axis=1) / dt
    mean_speed = np.mean(speeds) if len(speeds) > 0 else 0.0
    return mean_speed



def create_temp_video_dir(subset_images, temp_dir="temp_video_dir"):
    os.makedirs(temp_dir, exist_ok=True)
    
    for i, img in enumerate(subset_images):
        filename = f"{i:05d}.png"  # o .jpg a seconda del formato originale
        out_path = Path(temp_dir) / filename
        cv2.imwrite(str(out_path), img)

    return temp_dir