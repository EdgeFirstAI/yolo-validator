"""Shared OpenCV image load/decode for both preprocess paths.

Both the Torch and NumPy paths use this identical loader so the
load/decode stage is consistent and independently benchmarkable. It is
EXCLUDED from the apples-to-apples Ultralytics comparison (Ultralytics
does not measure image load/decode).
"""
from __future__ import annotations

import cv2
import numpy as np


def decode_image(path: str) -> tuple[np.ndarray, int, int]:
    """Load an image as BGR uint8. Returns (img, width, height)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not decode image: {path}")
    h, w = img.shape[:2]
    return img, int(w), int(h)
