"""NumPy class-aware non-maximum suppression.

Independent NumPy reimplementation (no Ultralytics source is used). Class
awareness applies the standard per-class box-offset technique — shift each
box by ``class_id * max_wh`` so boxes of different classes can never overlap,
then run a single greedy IoU suppression. ``max_wh = 7680`` is the
conventional YOLO offset constant. This technique was popularized by the
Ultralytics implementation; the algorithm is attributed, the code here is
original.

Note: this greedy suppression sorts by score descending and is order-stable
only for distinct scores. For exactly-equal scores the kept set may differ
from ``torchvision.ops.nms`` tie-breaking by a detection or two — a small,
bounded, documented difference vs the Ultralytics Torch path (see
BENCHMARK.md).
"""
from __future__ import annotations

import numpy as np


def _iou(box, others) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    area_o = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = area + area_o - inter
    return np.where(union > 0, inter / union, 0.0)


def nms(boxes, scores, iou_thres: float) -> np.ndarray:
    """Greedy IoU NMS. Returns kept indices, highest score first."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = _iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


def nms_class_aware(boxes, scores, classes, iou_thres: float, max_wh: float = 7680.0) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    offset = np.asarray(classes, dtype=np.float32).reshape(-1, 1) * max_wh
    return nms(boxes + offset, scores, iou_thres)
