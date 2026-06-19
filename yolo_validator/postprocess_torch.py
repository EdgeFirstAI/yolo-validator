# yolo_validator/postprocess_torch.py
"""Torch postprocess path — reuses Ultralytics ops for fidelity.

Uses ultralytics.utils.nms.non_max_suppression (moved from ops in 8.4.x)
so detections are numerically faithful to the Ultralytics validator.
Inputs and outputs match NumpyPostprocessor exactly (Detections in
original pixel coords) so the pipeline is path-agnostic. Mask
materialization is handled separately and shared with the NumPy path
(see masks.py).

Adaptation vs plan: ultralytics 8.4.67 moved non_max_suppression from
ultralytics.utils.ops to ultralytics.utils.nms. The call signature and
output column layout (N, 6+nm: xyxy, conf, cls, [coeffs]) are unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from ultralytics.utils.nms import non_max_suppression

from .backends import ModelSpec
from .detections import Detections
from .letterbox import LetterboxInfo, unletterbox_boxes


class TorchPostprocessor:
    def __init__(self, spec: ModelSpec, conf: float, iou: float, max_det: int):
        self.spec = spec
        self.conf = conf
        self.iou = iou
        self.max_det = max_det

    def decode(self, raw_outputs, lb: LetterboxInfo) -> Detections:
        pred = None
        protos = None
        for o in raw_outputs:
            if o.ndim == 4:
                protos = o
            else:
                pred = o
        nm = protos.shape[1] if (self.spec.task == "segment" and protos is not None) else 0
        nc = pred.shape[1] - 4 - nm
        pt = torch.from_numpy(np.asarray(pred, dtype=np.float32))
        out = non_max_suppression(
            pt, self.conf, self.iou, nc=nc, max_det=self.max_det,
            multi_label=True,  # match Ultralytics val default (DetectionValidator.postprocess)
        )[0]  # (N, 6+nm): xyxy, conf, cls, [coeffs]
        if out.shape[0] == 0:
            return Detections(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                              np.zeros((0,), np.int64),
                              None if nm == 0 else np.zeros((0, nm), np.float32), protos)
        boxes = out[:, :4].cpu().numpy()         # letterboxed input space
        scores = out[:, 4].cpu().numpy()
        classes = out[:, 5].cpu().numpy().astype(np.int64)
        coeffs = out[:, 6:].cpu().numpy() if nm else None
        boxes_orig = unletterbox_boxes(boxes, lb)
        return Detections(boxes_orig, scores, classes, coeffs, protos, boxes_lb=boxes)
