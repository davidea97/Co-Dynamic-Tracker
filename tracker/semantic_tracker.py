import torch
import sys
sys.path.append("./sam2")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import cv2
import numpy as np
import os
import random

class SemanticTracker:
    def __init__(self, window_len=8):
        self.checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = SAM2ImagePredictor(build_sam2(self.model_cfg, self.checkpoint))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    def mask_generator(self, image, tracks2d, output_dir=None, window_counter=0, image_counter=0):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.predictor.set_image(image_rgb)
                input_points = np.array(tracks2d)  # [[x1, y1], [x2, y2], ...]
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