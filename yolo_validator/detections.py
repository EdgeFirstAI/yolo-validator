# yolo_validator/detections.py
"""Canonical per-image detection result, backend- and path-agnostic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Detections:
    boxes: np.ndarray            # (N,4) xyxy in ORIGINAL image pixels
    scores: np.ndarray           # (N,)
    classes: np.ndarray          # (N,) int
    coeffs: Optional[np.ndarray] = None   # (N,32) mask coeffs (segment only)
    protos: Optional[np.ndarray] = None   # (1,32,mh,mw) (segment only)
    # xyxy in LETTERBOXED input pixels (pre-unletterbox). Ultralytics crops
    # masks at input resolution with these boxes (see masks.py); kept so the
    # NumPy mask path can mirror process_mask_native exactly.
    boxes_lb: Optional[np.ndarray] = None
