# yolo_validator/coco_output.py
"""COCO-format prediction serialization (bbox + segm RLE).

RLE via pycocotools.mask.encode (identical encoding to the Ultralytics
segmentation eval path). coco80->coco91 remap applied for COCO datasets.
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

from .detections import Detections


def coco80_to_coco91() -> list[int]:
    """80-class contiguous index -> 91-class COCO category_id.

    COCO-specific: the CLI applies this remap to every model, so a non-COCO
    80-class model would emit wrong `category_id`s. For COCO val2017 this is
    correct and required (the GT uses the sparse 91-id space).
    """
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
        85, 86, 87, 88, 89, 90,
    ]


def detections_to_coco(image_id: int, det: Detections, class_map, masks=None) -> list[dict]:
    # Resolve a GPU tensor to numpy once before the per-detection loop (single D2H).
    if masks is not None and len(masks):
        try:
            import torch
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()   # (N, H, W) uint8 — one transfer for all N
        except ImportError:
            pass
    recs: list[dict] = []
    for i in range(len(det.scores)):
        x0, y0, x1, y1 = (float(v) for v in det.boxes[i])
        rec = {
            "image_id": int(image_id),
            "category_id": int(class_map[int(det.classes[i])]),
            "bbox": [x0, y0, x1 - x0, y1 - y0],
            "score": float(det.scores[i]),
        }
        if masks is not None and i < len(masks):
            rle = mask_utils.encode(np.asfortranarray(masks[i].astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            rec["segmentation"] = rle
        recs.append(rec)
    return recs
