"""Shared letterbox geometry (used by both Torch and NumPy preprocess).

Center letterbox, aspect-preserving, gray pad. `scaleup=False` matches the
Ultralytics `val` default (small images are not upscaled). Exact parity
nuances vs Ultralytics LetterBox are tuned in Benchmark A; this is the
canonical center-pad math both paths share.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    scale: float
    pad_x: float
    pad_y: float
    orig_w: int
    orig_h: int


def compute_letterbox(orig_w, orig_h, input_w, input_h, scaleup: bool = False) -> LetterboxInfo:
    scale = min(input_w / orig_w, input_h / orig_h)
    if not scaleup:
        scale = min(scale, 1.0)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)
    pad_x = (input_w - new_w) / 2.0
    pad_y = (input_h - new_h) / 2.0
    return LetterboxInfo(float(scale), float(pad_x), float(pad_y), int(orig_w), int(orig_h))


def pad_int(lb: LetterboxInfo) -> tuple[int, int]:
    """Integer (left, top) pad used by Ultralytics LetterBox / scale_boxes.

    Ultralytics pads with ``round(pad - 0.1)`` (``LetterBox``) and un-pads with
    the same ``round((input - round(orig*gain)) / 2 - 0.1)`` (``scale_boxes``).
    Using the float ``pad`` here instead introduces a sub-pixel, systematic box
    shift, so box <-> original mapping must use this rounded pad.
    """
    return int(round(lb.pad_x - 0.1)), int(round(lb.pad_y - 0.1))


def unletterbox_boxes(boxes_xyxy, lb: LetterboxInfo) -> np.ndarray:
    """Map xyxy boxes from letterboxed input space to original pixel coords.

    Mirrors ``ultralytics.utils.ops.scale_boxes``: subtract the *rounded*
    integer pad, divide by gain, clip to the original image.
    """
    b = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
    if b.size == 0:
        return b
    pad_x, pad_y = pad_int(lb)
    b[:, [0, 2]] = (b[:, [0, 2]] - pad_x) / lb.scale
    b[:, [1, 3]] = (b[:, [1, 3]] - pad_y) / lb.scale
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, lb.orig_w)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, lb.orig_h)
    return b
