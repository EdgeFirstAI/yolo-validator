# yolo_validator/postprocess.py
"""Postprocess abstraction: decode raw model outputs -> Detections.

NumPy path implements the Ultralytics decode (transpose, conf filter,
class-aware NMS, xywh->xyxy, un-letterbox). Torch path (Task 10) reuses
Ultralytics ops. Mask materialization is a separate step (masks.py) the
pipeline calls only for segment models.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .backends import ModelSpec
from .detections import Detections
from .letterbox import LetterboxInfo, unletterbox_boxes
from .nms import nms_class_aware_cv2


def _xywh2xyxy(b: np.ndarray) -> np.ndarray:
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2
    out[:, 1] = b[:, 1] - b[:, 3] / 2
    out[:, 2] = b[:, 0] + b[:, 2] / 2
    out[:, 3] = b[:, 1] + b[:, 3] / 2
    return out


class Postprocessor(Protocol):
    def decode(self, raw_outputs: list[np.ndarray], lb: LetterboxInfo) -> Detections:
        ...


class NumpyPostprocessor:
    def __init__(self, spec: ModelSpec, conf: float, iou: float, max_det: int):
        self.spec = spec
        self.conf = conf
        self.iou = iou
        self.max_det = max_det

    def decode(self, raw_outputs, lb) -> Detections:
        pred = None
        protos = None
        for o in raw_outputs:
            if o.ndim == 4:
                protos = o
            else:
                pred = o
        p = np.squeeze(pred, 0).transpose(1, 0)  # (8400, C)
        # nm = mask-coefficient count = proto channel dim (read from the model,
        # not hardcoded). nc is then whatever channels remain after box + coeffs.
        nm = protos.shape[1] if (self.spec.task == "segment" and protos is not None) else 0
        nc = p.shape[1] - 4 - nm
        boxes_xywh = p[:, :4]
        cls_scores = p[:, 4 : 4 + nc]
        coeffs = p[:, 4 + nc :] if nm else None

        max_nms = 30000
        anc_idx, cls_idx = np.where(cls_scores > self.conf)
        if anc_idx.size == 0:
            return Detections(
                np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                np.zeros((0,), np.int64),
                None if nm == 0 else np.zeros((0, nm), np.float32),
                protos,
            )
        sc = cls_scores[anc_idx, cls_idx]
        # Cap to top max_nms by score
        if sc.size > max_nms:
            top = np.argpartition(sc, -max_nms)[-max_nms:]
            anc_idx, cls_idx, sc = anc_idx[top], cls_idx[top], sc[top]
        xyxy = _xywh2xyxy(boxes_xywh[anc_idx])
        co = coeffs[anc_idx] if nm else None
        # NMS via OpenCV (cv2.dnn, torch-free, in core deps): bit-accurate vs the
        # greedy reference (identical kept set) and 13-154x faster at conf=0.001
        # candidate counts. score_threshold=conf, top_k=max_det.
        idx = nms_class_aware_cv2(xyxy, sc, cls_idx, self.conf, self.iou, self.max_det)
        boxes_lb = xyxy[idx]                     # letterboxed input space (for masks)
        boxes_orig = unletterbox_boxes(boxes_lb, lb)
        return Detections(
            boxes=boxes_orig,
            scores=sc[idx],
            classes=cls_idx[idx].astype(np.int64),
            coeffs=(co[idx] if nm else None),
            protos=protos,
            boxes_lb=boxes_lb,
        )


def make_postprocessor(path: str, spec: ModelSpec, conf: float, iou: float, max_det: int) -> Postprocessor:
    if spec.e2e:
        from .postprocess_e2e import E2EPostprocessor
        return E2EPostprocessor(spec, conf, iou, max_det)
    if path in ("torch", "auto"):
        try:
            from .postprocess_torch import TorchPostprocessor

            return TorchPostprocessor(spec, conf, iou, max_det)
        except ImportError:
            if path == "torch":
                raise
    return NumpyPostprocessor(spec, conf, iou, max_det)
