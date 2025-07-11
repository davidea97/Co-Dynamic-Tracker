import cv2
import glob
import os

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