from cotracker.predictor import CoTrackerOnlinePredictor
import natsort
import glob
import os
import imageio.v3 as iio
import torch
import numpy as np
from tqdm import tqdm
from cotracker.utils.visualizer import Visualizer
import time
class OfflineTracker():
    def __init__(self, rgb_folder, intrinsics=None, poses_folder=None, grid_size=30, checkpoint="/home/allegro/davide_ws/co-tracker/checkpoints/scaled_online.pth", video_url=None):
        
        self.video_url = video_url
        self.rgb_folder = rgb_folder
        self.intrinsics = intrinsics
        self.poses_folder = poses_folder
        self.grid_size = grid_size
        self.checkpoint = checkpoint
        self.grid_query_frame = 0

        self.grid_size = grid_size

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

    def extract_image_files(self, folder):
        files_path = natsort.natsorted(glob.glob(os.path.join(folder, "*.png")))
        files = [iio.imread(file) for file in files_path]
        return files
    
    def extract_poses(self, folder):
        files_path = natsort.natsorted(glob.glob(os.path.join(folder, "*.npy")))
        files = [np.load(file) for file in files_path]
        return files

    def offline_tracking(self, window_len=8):
        # window_len = 8 #self.model.step
        frames = self.extract_image_files(self.rgb_folder) if self.video_url is None else list(iio.imiter(self.video_url, plugin="FFMPEG"))

        self.all_tracks = []
        self.all_visibilities = []
        self.query_frames = []

        self.global_tracks = []
        self.global_visibilities = []
        self.global_frames = []
        
        for start in tqdm(range(0, len(frames) - window_len + 1, window_len)):
            end = start + window_len
            window_frames = frames[start:end]
            # print(f"Processing frames {start} to {end} ({len(window_frames)} frames)")
            self._process_step(  # Inizializzazione
                window_frames,
                is_first_step=True,
                grid_size=self.grid_size,
                grid_query_frame=self.grid_query_frame,
            )

            pred_tracks, pred_visibility = self._process_step(  # Tracking
                window_frames,
                is_first_step=False,
                grid_size=self.grid_size,
                grid_query_frame=self.grid_query_frame,
            )

            self.global_tracks.append(pred_tracks[0])  # Keep only second half
            self.global_visibilities.append(pred_visibility[0])
            self.global_frames.extend(window_frames)

        remainder = len(frames) % window_len
        if remainder != 0:
            start = len(frames) - remainder
            window_frames = frames[start:]

            pred_tracks, pred_visibility = self._process_step(
                window_frames,
                is_first_step=False,
                grid_size=self.grid_size,
                grid_query_frame=self.grid_query_frame,
            )

            self.global_tracks.append(pred_tracks[0])
            self.global_visibilities.append(pred_visibility[0])
            self.global_frames.extend(window_frames)

        self.pred_tracks = torch.cat(self.global_tracks, dim=0)[None]  # (1, T, N, 2)
        self.pred_visibility = torch.cat(self.global_visibilities, dim=0)[None]

    def online_tracking(self, window_len=8):
        # window_len = 8 #self.model.step
        frames = self.extract_image_files(self.rgb_folder) if self.video_url is None else list(iio.imiter(self.video_url, plugin="FFMPEG"))

        self.all_tracks = []
        self.all_visibilities = []
        self.query_frames = []

        self.global_tracks = []
        self.global_visibilities = []
        # self.global_frames = []
        self.window_frames
        # is_first_step = True
        # for i in tqdm(range(0, len(frames) - window_len + 1)):
        counter = 0
        for i in tqdm(range(0, len(frames))):
            if i % window_len == 0 and i != 0:
                counter += 1
                # print("Processing window frames: ", len(self.window_frames))
                self._process_step(  # Inizializzazione
                    self.window_frames[i - window_len:i],
                    is_first_step=True,
                    grid_size=self.grid_size,
                    grid_query_frame=self.grid_query_frame,
                )

                pred_tracks, pred_visibility = self._process_step(  # Tracking
                    self.window_frames[i - window_len:i],
                    is_first_step=False,
                    grid_size=self.grid_size,
                    grid_query_frame=self.grid_query_frame,
                )

                self.global_tracks.append(pred_tracks[0])  # Keep only second half
                self.global_visibilities.append(pred_visibility[0])
                # self.global_frames.extend(self.window_frames)

            self.window_frames.append(frames[i])

        if len(self.window_frames) >= window_len and len(self.window_frames) % window_len == 0:
            self._process_step(
                self.window_frames[-window_len:],
                is_first_step=True,
                grid_size=self.grid_size,
                grid_query_frame=self.grid_query_frame,
            )
            pred_tracks, pred_visibility = self._process_step(
                self.window_frames[-window_len:],
                is_first_step=False,
                grid_size=self.grid_size,
                grid_query_frame=self.grid_query_frame,
            )
            self.global_tracks.append(pred_tracks[0])
            self.global_visibilities.append(pred_visibility[0])

        print("Counter: ", counter)
        print("Length of window frames: ", len(self.window_frames))
        remainder = len(self.window_frames) % window_len
        if remainder != 0:
            start = len(self.window_frames) - remainder
            # self.window_frames = frames[start:]

            pred_tracks, pred_visibility = self._process_step(
                self.window_frames[start:],
                is_first_step=False,
                grid_size=self.grid_size,
                grid_query_frame=self.grid_query_frame,
            )

            self.global_tracks.append(pred_tracks[0])
            self.global_visibilities.append(pred_visibility[0])
            # self.global_frames.extend(self.window_frames)

        self.pred_tracks = torch.cat(self.global_tracks, dim=0)[None]  # (1, T, N, 2)
        self.pred_visibility = torch.cat(self.global_visibilities, dim=0)[None]



    def save_video(self):
        video = torch.tensor(np.stack(self.window_frames), device=self.device).permute(
            0, 3, 1, 2
        )[None]
        vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
        vis.visualize(
            video, self.pred_tracks, self.pred_visibility, query_frame=self.grid_query_frame
        )
        print("Video saved successfully.")

    def save_online_video(self):
        video = torch.tensor(np.stack(self.global_frames), device=self.device).permute(
            0, 3, 1, 2
        )[None]
        vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
        vis.visualize(
            video, self.pred_tracks, self.pred_visibility, query_frame=self.grid_query_frame
        )
        print("Video saved successfully.")

