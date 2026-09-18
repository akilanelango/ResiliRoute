import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry


class ObstructionSegmenter:
    def __init__(self, checkpoint_path: str, model_type: str = "vit_b", device: str = None):
        """
        Initializes SAM with weights and hardware device (FR-SEG-2).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Initializing SAM ({model_type}) on device: {self.device}")

        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)

        # points_per_batch=16 limits peak VRAM usage to stay under 4GB on GTX 1050 Ti
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            points_per_batch=16,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=1000,
            crop_n_layers=0
        )
        self.predictor = SamPredictor(self.sam)

    def segment_automatic(self, image_rgb: np.ndarray, min_area_threshold: int = 2500) -> np.ndarray:
        """
        Runs automatic mask proposal and aggregates segments meeting the area threshold (FR-SEG-3, FR-SEG-4).
        Expects an 8-bit RGB image array: shape (H, W, 3), dtype uint8.
        """
        masks = self.mask_generator.generate(image_rgb)

        combined_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        for mask_data in masks:
            if mask_data["area"] >= min_area_threshold:
                combined_mask[mask_data["segmentation"]] = 255

        return combined_mask

    def segment_prompted(
        self,
        image_rgb: np.ndarray,
        input_points: np.ndarray,
        input_labels: np.ndarray
    ) -> np.ndarray:
        """
        Runs prompt-guided mask generation for pinpointed flood coordinates (FR-SEG-2).
        input_labels: 1 for foreground/obstruction, 0 for background.
        """
        self.predictor.set_image(image_rgb)
        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
        return (masks[0] * 255).astype(np.uint8)