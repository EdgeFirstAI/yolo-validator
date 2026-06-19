"""Direct fidelity of the NumPy mask path vs Ultralytics process_mask_native.

This compares yolo_validator.masks.materialize_masks_numpy against the *real*
Ultralytics validator mask pipeline (process_mask_native -> scale_masks ->
byte) on identical synthetic inputs (random protos/coeffs/boxes). It needs
torch + ultralytics; skipped otherwise. Unlike the yv-vs-yv mask-IoU check,
this is the cross-check against Ultralytics itself.
"""
from __future__ import annotations

import numpy as np
import pytest

from yolo_validator.letterbox import compute_letterbox
from yolo_validator.masks import materialize_masks_numpy


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


def test_numpy_masks_match_ultralytics_process_mask_native():
    torch = pytest.importorskip("torch")
    ops = pytest.importorskip("ultralytics.utils.ops")

    rng = np.random.default_rng(0)
    input_w = input_h = 640
    orig_w, orig_h = 640, 480           # letterbox pads top/bottom (pad_y=80)
    lb = compute_letterbox(orig_w, orig_h, input_w, input_h)
    c, mh, mw = 32, 160, 160
    n = 12

    protos = rng.standard_normal((1, c, mh, mw)).astype(np.float32)
    coeffs = rng.standard_normal((n, c)).astype(np.float32)
    # boxes in letterboxed input space, inside the valid (non-pad) region.
    x1 = rng.uniform(0, input_w - 60, n)
    y1 = rng.uniform(lb.pad_y, input_h - lb.pad_y - 60, n)
    boxes_lb = np.stack([x1, y1, x1 + rng.uniform(40, 120, n),
                         y1 + rng.uniform(40, 120, n)], axis=1).astype(np.float32)

    # --- Ultralytics reference (exactly as SegmentationValidator does it) ---
    proto_t = torch.from_numpy(protos[0])
    coeff_t = torch.from_numpy(coeffs)
    box_t = torch.from_numpy(boxes_lb)
    ref_640 = ops.process_mask_native(proto_t, coeff_t, box_t, shape=(input_h, input_w))
    ratio_pad = ((lb.scale, lb.scale), (lb.pad_x, lb.pad_y))
    ref_ori = ops.scale_masks(ref_640[None].float(), (orig_h, orig_w), ratio_pad=ratio_pad)[0].byte()
    ref = ref_ori.cpu().numpy()

    # --- yolo-validator NumPy path ---
    got = materialize_masks_numpy(protos, coeffs, boxes_lb, lb, input_w, input_h)

    assert len(got) == n
    ious = [_iou(got[i], ref[i]) for i in range(n)]
    mean_iou = float(np.mean(ious))
    # cv2 INTER_LINEAR vs torch bilinear differ only at edge pixels: expect
    # very high agreement, far above the pre-fix sigmoid/crop-order path.
    assert mean_iou > 0.97, f"mean mask IoU vs Ultralytics too low: {mean_iou:.4f} ({ious})"
