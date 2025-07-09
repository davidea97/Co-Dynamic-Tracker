import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from collections import defaultdict
import numpy as np
from numpy.linalg import svd
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import PCA

class DynamicTracker():
    def __init__(self, tracks, rgb_images, depth_images, camera_poses, intrinsics, start_frame=0, end_frame=None, tracker_step_window=8, grid_size=30):
        self.tracks = tracks
        self.rgb_images = rgb_images
        self.depth_images = depth_images
        self.camera_poses = camera_poses
        self.intrinsics = intrinsics
        self.start_frame = start_frame
        self.end_frame = end_frame if end_frame is not None else len(rgb_images)
        self.tracker_step_window = tracker_step_window
        self.grid_size = grid_size


    def extract_3D_points(self, save_detections=False):
        fx = self.intrinsics[0][0]  
        fy = self.intrinsics[1][1]
        cx = self.intrinsics[0][2]
        cy = self.intrinsics[1][2]
        if save_detections:
            output_path = "./detected_keypoints"
            os.makedirs(output_path, exist_ok=True)
            height, width = self.rgb_images[0].shape[:2]
            video_path = os.path.join(output_path, "detections_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # o 'XVID' se preferisci AVI
            out_video = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))

        points_3d_world = {}  # Dictionary to hold 3D points for each frame
        for t in range(self.start_frame, self.end_frame):
            
            if save_detections:
                rgb = self.rgb_images[t].copy()
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            pose = self.camera_poses[t]
            depth = self.depth_images[t]/1000

            all_depths = []
            keypoints = []
            for n in range(self.tracks.shape[1]):
                x, y = self.tracks[t, n]
                x, y = int(x), int(y)
                if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
                    z = depth[y, x]
                    if z > 0:
                        all_depths.append(z)

            for n in range(self.tracks.shape[1]):
                x, y = self.tracks[t, n]
                x, y = int(x), int(y)

                if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
                    z = depth[y, x]   ## meters
                    if z == 0:
                        continue

                    if save_detections:
                        b=0
                        g=255
                        r=0
                        color = (b, g, r)

                        cv2.circle(rgb, (x, y), radius=2, color=color, thickness=-1)
                    
                    X = (x - cx) * z / fx
                    Y = (y - cy) * z / fy
                    Z = z 
                    
                    cam_coords = np.array([X, Y, Z])
                    cam_coords_hom = np.append(cam_coords, 1.0) 
                    world_coords = pose @ cam_coords_hom   # 4D world point
                    keypoints.append((n, world_coords[:3])) 
            
            points_3d_world[t] = keypoints

            # Save frame with keypoints
            if save_detections:
                out_video.write(rgb)
                cv2.imwrite(os.path.join(output_path, f"frame_{t:04d}.png"), rgb)
        if save_detections:
            out_video.release()
        return points_3d_world

    def compute_track_to_grid_index(self, grid_size):
        """
        Costruisce una mappa {n: (i, j)} che associa ogni punto track ID n a una cella (i, j)
        della griglia grid_size x grid_size calcolata sull'immagine.
        """
        track_to_grid_index = defaultdict(dict)  # window_id -> {n: (i, j)}

        height, width = self.rgb_images[0].shape[:2]
        cell_h = height / grid_size
        cell_w = width / grid_size

        for t in range(self.start_frame, self.end_frame):
            relative_index = t - self.start_frame
            window_id = relative_index // self.tracker_step_window

            if relative_index % self.tracker_step_window == 0:  # primo frame della finestra
                for n in range(self.tracks.shape[1]):
                    x, y = self.tracks[t, n]
                    i = int(y // cell_h)
                    j = int(x // cell_w)
                    i = min(max(i, 0), grid_size - 1)
                    j = min(max(j, 0), grid_size - 1)
                    track_to_grid_index[window_id][n] = (i, j)

        return track_to_grid_index
    
    def is_dynamic_pca(self, track_array, threshold_energy=0.95):
        """
        Given a track_array of shape (N, 3), determine if the point is dynamic using PCA.
        """
        if len(track_array) < 3:
            return False

        track_T = track_array.T  # shape: (3, N)
        U, S, Vt = svd(track_T - np.mean(track_T, axis=1, keepdims=True), full_matrices=False)
        energy = np.cumsum(S**2) / np.sum(S**2)

        return energy[0] < threshold_energy or (len(energy) > 1 and energy[1] > 0.05)


    def is_dynamic_spline(self, track_array):
        """
        Given a track_array of shape (N, 3), determine if the point is dynamic using spline fitting.
        """
        if len(track_array) < 4:
            return False

        t_vals = np.linspace(0, 1, len(track_array))
        residuals = []

        for i in range(3):
            try:
                spline = UnivariateSpline(t_vals, track_array[:, i], s=0.1)
                fitted = spline(t_vals)
                residuals.append(np.mean((fitted - track_array[:, i])**2))
            except Exception:
                return False  # fallback to static if fitting fails

        total_residual = np.sum(residuals)
        return total_residual > 1e-4  # This could be replaced by a dynamic thresholding strategy


    
    def extract_dynamic_points(self, points_3d_world, track_grid):
        
        """
        points_3d_world: {t: [(n, point_3d)]}
        """
        track_3d = defaultdict(list)
        frame_map = defaultdict(list)

        for t in range(self.start_frame, self.end_frame):
            relative_index = t - self.start_frame
            window_id = relative_index // self.tracker_step_window

            for n, point in points_3d_world.get(t, []):
                track_3d[(window_id, n)].append(point)
                frame_map[(window_id, n)].append(t)

        # Output per frame
        per_frame_static = defaultdict(list)
        per_frame_dynamic = defaultdict(list)
        per_frame_outliers = defaultdict(list)
        raw_dynamic_by_window = defaultdict(list)

        # SINGULAR TEMPORAL CHECKING
        for (window_id, n), track in track_3d.items():
            
            track_array = np.array(track)
            center = np.median(track_array, axis=0)
            spread = np.median(np.linalg.norm(track_array - center, axis=1))

            # JUMP CHECKING
            # If the track is too short, treat it as static
            diffs = np.linalg.norm(np.diff(track_array, axis=0), axis=1)
            if len(diffs) < 2:
                for point, t in zip(track, frame_map[(window_id, n)]):
                    per_frame_static[t].append((n, point, spread))
                continue

            max_jump = np.max(diffs)
            jump_threshold = 0.05
            if max_jump > 2 * jump_threshold:
                # Salto anomalo rilevato: probabile outlier -> trattalo come statico
                is_dynamic = False
            else:
                is_dynamic = spread > 0.03  # soglia dinamica fissa, puoi cambiarla o usare una soglia robusta  
                # is_dynamic = self.is_dynamic_pca(track_array) 
                # is_dynamic = self.is_dynamic_spline(track_array)

            for point, t in zip(track, frame_map[(window_id, n)]):
                if is_dynamic:
                    raw_dynamic_by_window[window_id].append((n, point, spread, t))
                    # per_frame_dynamic[t].append((n, point, spread))
                else:
                    per_frame_static[t].append((n, point, spread))

        # SPATIAL TRAJECTORY CHECKING
        counter = 0

        for window_id, dynamic_list in raw_dynamic_by_window.items():
            grid = defaultdict(list)  # (i, j) -> list of entries
            counter += 1
            for entry in dynamic_list:
                n, point, spread, t = entry
                if n not in track_grid.get(window_id, {}):
                    continue  # punto non presente nella griglia
                i, j = track_grid[window_id][n]

                grid[(i, j)].append(entry)

            valid_cells, dynamic_cluster = self.extract_dynamic_clusters(grid, min_cluster_size=7)

            from itertools import combinations

            for cluster_cells in dynamic_cluster:
                cluster_entries = []

                # Raccogli tutti gli entry (n, point, spread, t) per tutte le celle del cluster
                for (i, j) in cluster_cells:
                    cluster_entries.extend(grid[(i, j)])

                # Mappa da n -> lista di punti 3D nel tempo (la traiettoria)
                cluster_tracks = defaultdict(list)
                for n, point, spread, t in cluster_entries:
                    cluster_tracks[n].append((t, point))

                # Deviazione media per punto
                # Raggruppa per frame: t -> {n: point}
                points_by_time = defaultdict(dict)
                for n, point, spread, t in cluster_entries:
                    points_by_time[t][n] = point

                # Estrai lista ordinata di frame comuni a tutti
                common_times = sorted(points_by_time.keys())
                if len(common_times) < 2:
                    continue

                # Calcola le distanze tra tutte le coppie nel primo frame
                ref_time = common_times[0]
                ref_points = points_by_time[ref_time]
                ref_pairs = list(combinations(ref_points.keys(), 2))
                ref_distances = {}
                for n1, n2 in ref_pairs:
                    p1, p2 = ref_points[n1], ref_points[n2]
                    ref_distances[(n1, n2)] = np.linalg.norm(p1 - p2)

                # Verifica la variazione delle stesse distanze nel tempo
                variations = []
                for t in common_times[1:]:
                    pts = points_by_time[t]
                    for (n1, n2), ref_d in ref_distances.items():
                        if n1 in pts and n2 in pts:
                            d = np.linalg.norm(pts[n1] - pts[n2])
                            variations.append(abs(d - ref_d))

                # Calcola la deviazione media
                if not variations:
                    continue
                mean_dist_var = np.median(variations)
                print(f"Window {window_id}, cluster size {len(cluster_entries)}: mean distance variation = {mean_dist_var:.4f}")
                if mean_dist_var < 0.03:  # soglia empirica da tarare
                    for n, point, spread, t in cluster_entries:
                        per_frame_dynamic[t].append((n, point, spread))
                else:
                    for n, point, spread, t in cluster_entries:
                        per_frame_static[t].append((n, point, spread))

        print("Counter of dynamic clusters:", counter)
        return per_frame_static, per_frame_dynamic

    def trajectory_similarity(self, trajs, max_mse=0.01):
        """
        Input: list of N trajectories, each of shape (T, 3)
        Output: True if trajectories are consistent (i.e. similar), else False
        """
        if len(trajs) < 2:
            return False

        # Allineamento temporale grezzo
        min_len = min(len(t) for t in trajs)
        trajs = [t[:min_len] for t in trajs]

        traj_stack = np.stack(trajs)  # (N, T, 3)
        mean_traj = np.mean(traj_stack, axis=0)  # (T, 3)

        # Errore medio tra ogni traiettoria e la media
        errors = [np.mean(np.linalg.norm(traj - mean_traj, axis=1)) for traj in traj_stack]
        mean_error = np.mean(errors)

        return mean_error < max_mse
    

    def extract_dynamic_clusters(self, grid, min_cluster_size=5):
        from collections import deque

        visited = set()
        output_clusters = []
        all_clusters = []

        for cell in grid.keys():
            if cell in visited:
                continue

            # Nuovo cluster
            cluster = []
            queue = deque([cell])
            visited.add(cell)

            while queue:
                i, j = queue.popleft()
                cluster.append((i, j))

                # Visita tutti i vicini (8-connected)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if (di == 0 and dj == 0):
                            continue
                        neighbor = (ni, nj)
                        if neighbor in grid and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

            if len(cluster) >= min_cluster_size:
                output_clusters.extend(cluster)
                all_clusters.append(cluster)

        # Restituisce solo le celle dei cluster validi
        return set(output_clusters), all_clusters

    def get_dynamic_points_2d(self, per_frame_dynamic):
        from collections import defaultdict
        dynamic_2d_points_per_frame = defaultdict(list)

        for t, point_list in per_frame_dynamic.items():
            for n, _, _ in point_list:  # (n, point_3d, spread)
                if n >= self.tracks.shape[1]:
                    continue
                x, y = self.tracks[t, n]
                dynamic_2d_points_per_frame[t].append((int(x), int(y)))

        return dynamic_2d_points_per_frame

    def draw_dynamic_points(self, per_frame_static, per_frame_dynamic, output_path=None):
        import cv2, os
        os.makedirs(output_path, exist_ok=True)
        dynamic_frames = []

        for t in range(self.start_frame, self.end_frame):
            frame = self.rgb_images[t].copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Dinamici = rossi
            for n, _, _ in per_frame_dynamic.get(t, []):
                x, y = self.tracks[t, n]
                cv2.circle(frame, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)

            # Statici = verdi
            for n, _, _ in per_frame_static.get(t, []):
                x, y = self.tracks[t, n]
                cv2.circle(frame, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)

            dynamic_frames.append(frame)
            # cv2.imwrite(os.path.join(output_path, f"dyn_frame_{t:04d}.png"), frame)

        # Salva video
        height, width, _ = self.rgb_images[0].shape
        video_path = os.path.join(output_path, "dynamic_points_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
        for frame in dynamic_frames:
            video_writer.write(frame)
        video_writer.release()
        print(f"Video salvato in {video_path}")

    def draw_points_by_spread(self, per_frame_static, per_frame_dynamic, output_path=None):
        import cv2, os
        import numpy as np
        from collections import defaultdict

        os.makedirs(output_path, exist_ok=True)
        dynamic_frames = []

        # 1. Estrai tutti gli spread per calcolare lo spread massimo
        all_spreads = []
        for frame_data in list(per_frame_static.values()) + list(per_frame_dynamic.values()):
            for _, _, spread in frame_data:
                all_spreads.append(spread)

        if not all_spreads:
            print("Nessun punto da visualizzare.")
            return

        all_spreads = np.array(all_spreads)
        max_spread = np.percentile(all_spreads, 95)  # soglia massima robusta per evitare outlier
        min_spread = np.min(all_spreads)

        print(f"Color mapping: min spread = {min_spread:.4f}, max spread (95° perc) = {max_spread:.4f}")

        def spread_to_color(spread):
            """
            Mappa lo spread in un colore da verde (statico) a rosso (dinamico):
            - basso spread = verde (0,255,0)
            - medio = giallo (0,255,255)
            - alto spread = rosso (0,0,255)
            """
            normalized = np.clip((spread - min_spread) / (max_spread - min_spread), 0, 1)
            r = int(255 * normalized)
            g = int(255 * (1 - normalized))
            b = 0
            return (b, g, r)

        for t in range(self.start_frame, self.end_frame):
            frame = self.rgb_images[t].copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Disegna dinamici
            for n, _, spread in per_frame_dynamic.get(t, []):
                x, y = self.tracks[t, n]
                color = spread_to_color(spread)
                cv2.circle(frame, (int(round(x)), int(round(y))), 2, color, -1)

            # Disegna statici
            for n, _, spread in per_frame_static.get(t, []):
                x, y = self.tracks[t, n]
                color = spread_to_color(spread)
                cv2.circle(frame, (int(round(x)), int(round(y))), 2, color, -1)

            dynamic_frames.append(frame)
            cv2.imwrite(os.path.join(output_path, f"spread_frame_{t:04d}.png"), frame)

        # Salva video
        height, width, _ = self.rgb_images[0].shape
        video_path = os.path.join(output_path, "spread_points_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
        for frame in dynamic_frames:
            video_writer.write(frame)
        video_writer.release()
        print(f"Video salvato in {video_path}")