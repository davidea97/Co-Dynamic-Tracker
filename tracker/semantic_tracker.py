import torch
import sys
sys.path.append("./sam2")
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import cv2
import numpy as np
import os
import random
from tracker.utils.general_utils import create_temp_video_dir
import shutil
import time

class SemanticTracker:
    def __init__(self, window_len=8):
        self.checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.predictor = SAM2ImagePredictor(build_sam2(self.model_cfg, self.checkpoint))
        self.video_predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint, device=self.device)

        
        self.mask_autmatic_generator = SAM2AutomaticMaskGenerator(build_sam2(self.model_cfg, self.checkpoint, device=self.device, apply_postprocessing=False))
        self.window_len = window_len

    def automatic_mask_generator(self, image, output_dir=None, window_counter=0, image_counter=0):
        masks = self.mask_autmatic_generator.generate(image)
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            colored_mask = np.zeros_like(image)
            for i, mask in enumerate(masks):
                color = [random.randint(0, 255) for _ in range(3)]

                segmentation = mask['segmentation'].astype(bool)
                colored_mask[segmentation] = color

            mask_path = os.path.join(output_dir, f"mask_{window_counter*self.window_len+image_counter:04d}.png")
            cv2.imwrite(mask_path, colored_mask)

    def mask_generator(self, image, dynamic_points, static_points=None, output_dir=None, window_counter=0, image_counter=0):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image_rgb)
            if static_points is not None:
                input_points = np.array(dynamic_points + static_points) 
                input_labels = np.array([1] * len(dynamic_points) + [0] * len(static_points), dtype=np.int32)
            else:
                input_points = np.array(dynamic_points)  # [[x1, y1], [x2, y2], ...]
                input_labels = np.ones(len(input_points), dtype=np.int32)  # tutti foreground

            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
            )
            
        mask_arrays = []
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            for i, mask in enumerate(masks):
                mask_np = mask.astype("uint8") * 255
                out_path = os.path.join(output_dir, f"mask_frame_{window_counter*self.window_len+image_counter:04d}.png")
                cv2.imwrite(out_path, mask_np)
                mask_arrays.append(mask.astype(bool))

        return mask_arrays
    
    def window_mask_generator(self, rgb_images, tracks2d, window_counter, output_dir=None, verbose=True):
        """
        Generate masks for a sequence of images based on the provided 2D tracks.
        """
        temp_video_dir = create_temp_video_dir(rgb_images)
        # print(f"Temporary video directory created in {time.time() - time_dir_creation:.2f} seconds")
        inference_state = self.video_predictor.init_state(video_path=temp_video_dir)
        # print(f"Inference state initialized in {time.time() - time_inference_state:.2f} seconds")

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        input_points = np.array(tracks2d)
        input_labels = np.ones(len(input_points), dtype=np.int32)

        time_add_new_points = time.time()
        _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=input_points,
            labels=input_labels,
        )
        # print(f"New points added in {time.time() - time_add_new_points:.2f} seconds")
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        image_counter = 0
        mask_arrays = [None] * len(rgb_images)
        time_propagate_in_video = time.time()
        for _, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
            for i, _ in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().astype(np.float32)[0]

                mask_np = mask.astype("uint8") * 255
                if verbose:
                    out_path = os.path.join(output_dir, f"mask_frame_{window_counter*self.window_len+image_counter:04d}.png")
                    cv2.imwrite(out_path, mask_np)
            
            mask_arrays[image_counter] = mask.astype(bool)
            image_counter += 1
        # print(f"Propagation in video completed in {time.time() - time_propagate_in_video:.2f} seconds")
        shutil.rmtree(temp_video_dir)
        self.video_predictor.reset_state(inference_state)
        return mask_arrays

