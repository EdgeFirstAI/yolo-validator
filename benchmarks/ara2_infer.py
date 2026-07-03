"""On-device Ara240 (Kinara Ara-2 NPU) dvapi inference + COCO scoring.

Reference lane for ``imx95-ara240``: vendor pre-compiled ``.dvm`` models run
through the vendor host stack — OpenCV letterbox → int8 quantize → dvapi
``infer_sync`` (ctypes over libaraclient via the dvproxy Unix socket) →
dequantize → the same shared NumPy decode the other lanes use.

Because libaraclient aborts (glibc double free) after roughly 100–200
sequential inferences in one process, the orchestrator runs the val set in
**chunked subprocesses** (``--chunk``, default 100, retried once), merges the
partial predictions/timings, and scores once with ``canonical_eval``
(crowd-as-normal). Per frame it records the wall-clock inference roundtrip
plus the driver-reported sub-timings: DMA host→device, **core NPU compute**,
DMA device→host (dvapi reports them in µs despite its struct comments saying
ms). ``pre`` = letterbox+quantize (JPEG decode is reported separately under
``stats`` only); ``post`` = dequant+NMS+masks+COCO/RLE (same as the tflite
lane); ``e2e`` = pre+inf+post; single-stream, so stages are additive.

Emits ``benchmark_a_<model>_<ts>.json`` (config key ``yv-ara2``) that
``benchmarks.normalize`` folds into ``metrics/imx95-ara240.json``. The
``--workflow-vendor`` label records artifact provenance per row
(``kinara-sdk-1.2.1`` local DVMs vs ``nxp-hf-2.0.4`` from HF ``nxp/YOLOv8``).

Usage (on the target; scoring can be deferred to the host with --no-eval)::

    python3 -m benchmarks.ara2_infer --dvm yolov8n-seg-kinara-1.2.1.dvm \
        --model yolov8n-seg --workflow-vendor kinara-sdk-1.2.1 \
        --coco-val ~/coco/val2017 \
        --gt ~/coco/annotations/instances_val2017.json --no-eval

    # host: score a merged file produced by --no-eval
    python -m benchmarks.ara2_infer --finalize merged_yolov8n-seg_x.json \
        --gt ~/coco/annotations/instances_val2017.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from yolo_validator.coco_output import coco80_to_coco91, detections_to_coco
from yolo_validator.detections import Detections
from yolo_validator.letterbox import LetterboxInfo, unletterbox_boxes
from yolo_validator.masks import materialize_masks_numpy
from yolo_validator.nms import nms_class_aware

_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# Per-frame stages (ms). npu_* are dvproxy driver stats (µs→ms), NaN when the
# driver returns no stats for a frame.
STAGE_KEYS = ("decode", "preprocess", "inference", "postprocess", "e2e",
              "npu_h2d", "npu_core", "npu_d2h")


def _safe_label(name: str) -> str:
    """Reject model labels with path/shell metacharacters before they reach a
    filename (defense-in-depth for untrusted CLI input)."""
    if not _LABEL_RE.fullmatch(name) or ".." in name:
        raise SystemExit(f"invalid model label {name!r}: expected [A-Za-z0-9._-]")
    return name


def _stats(vals) -> dict | None:
    """min/mean/p50/p95/p99/max/n over NaN-filtered vals — NO trimming."""
    a = np.asarray([v for v in vals if not math.isnan(v)], dtype=np.float64)
    if a.size == 0:
        return None
    return {"min": float(a.min()), "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()), "n": int(a.size)}


def _mean(vals) -> float | None:
    s = _stats(vals)
    return s["mean"] if s else None


def _strip_trailing_ones(shape: list[int]) -> list[int]:
    """Strip trailing size-1 dims (Ara240 shape padding convention)."""
    shape = list(shape)
    while len(shape) > 2 and shape[-1] == 1:
        shape = shape[:-1]
    return shape


def _infer_role(shape, seen: set) -> str:
    """Guess an output tensor's role from its (channels-first) shape."""
    s = _strip_trailing_ones(list(shape))
    if len(s) == 3 and 32 in s and max(s) > 100:
        return "protos"
    if len(s) >= 2:
        feat = min(s[0], s[-1]) if len(s) == 2 else s[0]
        if feat == 4:
            return "boxes"
        if feat == 2:  # split decoder: first (2,N) is box_xy, second box_wh
            return "box_wh" if "box_xy" in seen else "box_xy"
        if feat == 80:
            return "scores"
        if feat == 32:
            return "mask_coefs"
    return "unknown"


def _to_nc(a: np.ndarray) -> np.ndarray:
    """(1,C,N) | (C,N) | (N,C) → (N,C)."""
    if a.ndim == 3:
        a = a[0]
    # Transpose if C < N, or if first dim matches known channel sizes (2,4,32,80)
    if a.ndim == 2 and (a.shape[0] < a.shape[1] or a.shape[0] in (2, 4, 32, 80)):
        return a.T
    return a


def _decode_outputs(outputs, imgsz, lb, image_id, class_map, score_th,
                    iou=0.7, max_det=300, top_k=1000):
    """Dequantized role tensors → COCO prediction dicts (original coords).

    outputs: "boxes" (4,N) xywh letterbox px — or "box_xy"+"box_wh" (2,N)
    each — plus "scores" (nc,N) in-graph-sigmoid'd probs, and for seg
    "mask_coefs" (32,N) + "protos" (32,ph,pw).
    """
    if "boxes" in outputs:
        boxes = _to_nc(outputs["boxes"])                      # (N,4) xywh
    else:
        boxes = np.concatenate(
            [_to_nc(outputs["box_xy"]), _to_nc(outputs["box_wh"])], axis=1)
    scores = _to_nc(outputs["scores"])                        # (N,nc)
    cx, cy, w, h = boxes.T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    flat = scores.reshape(-1)
    keep = np.nonzero(flat > score_th)[0]
    if not keep.size:
        return []
    if keep.size > top_k:
        keep = keep[np.argpartition(flat[keep], -top_k)[-top_k:]]
    nc = scores.shape[1]
    anc, cl, sc = keep // nc, keep % nc, flat[keep]
    box_lb = xyxy[anc]
    idx = nms_class_aware(box_lb, sc, cl, iou)[:max_det]
    box_lb, cls, sc = box_lb[idx], cl[idx].astype(np.int64), sc[idx]
    if "protos" in outputs and "mask_coefs" in outputs:
        coeffs = _to_nc(outputs["mask_coefs"])[anc][idx]
        proto = outputs["protos"]
        if proto.ndim == 4:
            proto = proto[0]                                  # (32,ph,pw)
        det = Detections(boxes=unletterbox_boxes(box_lb, lb), scores=sc,
                         classes=cls, coeffs=coeffs,
                         protos=proto[None].astype(np.float32),
                         boxes_lb=box_lb)
        masks = materialize_masks_numpy(det.protos, det.coeffs, det.boxes_lb,
                                        lb, imgsz, imgsz)
        return detections_to_coco(image_id, det, class_map, masks)
    det = Detections(boxes=unletterbox_boxes(box_lb, lb), scores=sc,
                     classes=cls)
    return detections_to_coco(image_id, det, class_map, None)
