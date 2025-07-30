import cv2
import glob
import os
import numpy as np

def make_video_from_frames(frame_dir, output_path, fps=10):
    output_folder = "output_video"
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted(glob.glob(f"{frame_dir}/frame_*.png"))
    if not image_files:
        print("Nessun frame trovato")
        return

    # Leggi la dimensione dal primo frame
    frame = cv2.imread(image_files[0])
    height, width, _ = frame.shape

    writer = cv2.VideoWriter(
        os.path.join(output_folder, output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    for img_path in image_files:
        frame = cv2.imread(img_path)
        writer.write(frame)

    writer.release()

def save_init_dynamic_estimation( window_rgb_images, pred_tracks, per_frame_dynamic, per_frame_static, output_dir="output_visualization", window_counter=0, window_len=8):

    os.makedirs(output_dir, exist_ok=True)
    tracks_2d = pred_tracks[0].cpu().numpy()  # [T, N, 2]
    for t, img in enumerate(window_rgb_images):
        img_out = img.copy()
        frame_idx = t

        # Disegna i punti statici in verde
        for (n, _, _, _) in per_frame_static.get(frame_idx, []):
            x, y = tracks_2d[t, n]
            cv2.circle(img_out, (int(x), int(y)), 2, (0, 255, 0), -1)  # Verde

        # Disegna i punti dinamici in rosso
        for (n, _, _, _, _) in per_frame_dynamic.get(frame_idx, []):
            x, y = tracks_2d[t, n]
            cv2.circle(img_out, (int(x), int(y)), 2, (255, 0, 0), -1)  # Rosso

        filename = os.path.join(output_dir, f"frame_{window_counter*window_len + t:04d}.png")
        cv2.imwrite(filename, cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))


def save_refined_dynamic_visualization(window_rgb_images, pred_tracks, refined_2d_points, output_dir="output_refined_visualization", window_counter=0, window_len=8):
    """
    Save refined dynamic points visualization.
    """
    os.makedirs(output_dir, exist_ok=True)
    tracks_2d = pred_tracks[0].cpu().numpy()  # [T, N, 2]

    for t, img in enumerate(window_rgb_images):
        img_out = img.copy()
        frame_idx = t

        # Dynamic point (red)
        refined_pts = refined_2d_points[t]
        for _, pt in refined_pts:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(img_out, (x, y), 2, (255, 0, 0), -1)  # Red

        # Static point (green)
        for n in range(tracks_2d.shape[1]):
            x, y = tracks_2d[t, n]
            if not any(np.allclose([x, y], rp[1], atol=1.5) for rp in refined_pts):
                cv2.circle(img_out, (int(x), int(y)), 2, (0, 255, 0), -1)  # Verde

        filename = os.path.join(output_dir, f"frame_{window_counter*window_len + t:04d}.png")
        cv2.imwrite(filename, cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))