# yolo_validator/postprocess_e2e.py
"""E2E (NMS-free) postprocessor for yolo26 end-to-end export.

Decodes [1, N, 6(+nm)] output where columns are [x1, y1, x2, y2, score, class_id, (nm coeffs)].
Coordinates are in letterboxed input pixel space [0, input_size] — verified empirically
by running yolo26n E2E export on COCO val images. No NMS is applied (the model already
ran NMS internally). Filters by score > conf, takes top max_det, then un-letterboxes.
"""
from __future__ import annotations

import numpy as np

from .backends import ModelSpec
from .detections import Detections
from .letterbox import LetterboxInfo, unletterbox_boxes


class E2EPostprocessor:
    """Decode [1, N, 6(+nm)] E2E ONNX output to Detections.

    Column layout (verified): x1, y1, x2, y2, score, class_id, [nm mask coeffs]
    Coordinate space: letterboxed input pixel space [0, input_size].
    """

    def __init__(self, spec: ModelSpec, conf: float, iou: float, max_det: int):
        self.spec = spec
        self.conf = conf
        self.max_det = max_det
        # iou not used (model already ran NMS), stored for API parity
        self.iou = iou

    def decode(self, raw_outputs: list[np.ndarray], lb: LetterboxInfo) -> Detections:
        pred = None
        protos = None
        for o in raw_outputs:
            if o.ndim == 4:
                protos = o
            else:
                pred = o

        nm = protos.shape[1] if (self.spec.task == "segment" and protos is not None) else 0
        rows = pred[0]  # (N, 6+nm): x1,y1,x2,y2,score,class,[coeffs]

        keep = rows[:, 4] > self.conf
        rows = rows[keep]

        if rows.shape[0] == 0:
            return Detections(
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.int64),
                None if nm == 0 else np.zeros((0, nm), np.float32),
                protos,
            )

        # Sort by score descending, take top max_det
        order = np.argsort(rows[:, 4])[::-1][: self.max_det]
        rows = rows[order]

        boxes = rows[:, :4].astype(np.float32)   # letterboxed input space
        scores = rows[:, 4].astype(np.float32)
        classes = rows[:, 5].astype(np.int64)
        coeffs = rows[:, 6 : 6 + nm].astype(np.float32) if nm else None

        boxes_orig = unletterbox_boxes(boxes, lb)
        return Detections(boxes_orig, scores, classes, coeffs, protos, boxes_lb=boxes)
