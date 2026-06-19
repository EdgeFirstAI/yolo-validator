"""Preprocess abstraction with two interchangeable paths.

- NumpyPreprocessor: OpenCV letterbox + BGR->RGB + /255 -> NCHW float32.
- TorchPreprocessor: mirrors Ultralytics (added in Task 7); falls back to
  numpy when torch/ultralytics are unavailable.

Both consume the shared OpenCV-decoded BGR image and the shared letterbox
geometry, so only the implementation (not the result) differs.
"""
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from .letterbox import LetterboxInfo, compute_letterbox

PAD_VALUE = 114


class Preprocessor(Protocol):
    def preprocess(self, img_bgr: np.ndarray, orig_w: int, orig_h: int) -> tuple[np.ndarray, LetterboxInfo]:
        ...


class NumpyPreprocessor:
    def __init__(self, input_w: int, input_h: int, scaleup: bool = False):
        self.input_w = input_w
        self.input_h = input_h
        self.scaleup = scaleup

    def preprocess(self, img_bgr, orig_w, orig_h):
        lb = compute_letterbox(orig_w, orig_h, self.input_w, self.input_h, self.scaleup)
        new_w = round(orig_w * lb.scale)
        new_h = round(orig_h * lb.scale)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top = int(round(lb.pad_y - 0.1))
        bottom = int(round(lb.pad_y + 0.1))
        left = int(round(lb.pad_x - 0.1))
        right = int(round(lb.pad_x + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
        )
        # ensure exact target size (rounding guard)
        padded = padded[: self.input_h, : self.input_w]
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return chw[None, ...], lb


def make_preprocessor(path: str, input_w: int, input_h: int, scaleup: bool = False) -> Preprocessor:
    """path: 'numpy' | 'torch' | 'auto'. 'auto' prefers torch when importable."""
    if path in ("torch", "auto"):
        try:
            from .preprocess_torch import TorchPreprocessor

            return TorchPreprocessor(input_w, input_h, scaleup)
        except ImportError:
            if path == "torch":
                raise
    return NumpyPreprocessor(input_w, input_h, scaleup)
