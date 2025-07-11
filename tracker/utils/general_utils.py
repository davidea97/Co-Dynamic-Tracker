import natsort
import glob
import imageio.v3 as iio
import os
import numpy as np
import cv2

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




