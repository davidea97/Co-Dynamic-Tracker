import natsort
import glob
import imageio.v3 as iio
import os
import numpy as np

def extract_image_files(folder):
        files_path = natsort.natsorted(glob.glob(os.path.join(folder, "*.png")))
        files = [iio.imread(file) for file in files_path]
        return files
    
def extract_poses(folder):
    files_path = natsort.natsorted(glob.glob(os.path.join(folder, "*.npy")))
    files = [np.load(file) for file in files_path]
    return files