# yolo_validator/masks.py
"""Mask materialization (segment models).

Faithful NumPy port of the Ultralytics validator's *native* mask pipeline
(``process_mask_native`` + ``scale_masks`` + ``crop_mask``, as used by
``SegmentationValidator`` when ``save_json=True``):

    raw logits = coeffs @ proto            # (N, mh, mw), NO sigmoid
    -> resize proto -> input size          # scale_masks to imgsz (square)
    -> crop to bbox at input res           # crop_mask, letterboxed boxes, float bounds
    -> threshold raw logits > 0            # .gt_(0.0) -> binary at input res
    -> strip letterbox pad + resize -> orig# scale_masks to ori_shape
    -> floor (truncate) -> binary at orig  # .byte()

The threshold is applied to **raw logits at input resolution** (not to
sigmoid probabilities after resizing), the crop happens at input resolution
with the **letterboxed** boxes (float bounds), and the final original-size
mask is the floor of the bilinearly-resized binary mask. These three details
are what make the output match Ultralytics; the previous implementation
sigmoided before resizing, cropped at original resolution, and thresholded
after both resizes, which produced a small systematic mask-mAP offset.

Fidelity: measured **bit-identical** to Ultralytics on COCO geometries
(mean mask-IoU 1.00000, 0% pixel disagreement vs ``process_mask_native`` +
``scale_masks`` over random protos/coeffs/boxes; see
``tests/test_masks_fidelity.py``). In principle OpenCV ``INTER_LINEAR`` and
Torch bilinear could differ by an edge pixel, but in practice they agree
here, so the previous systematic mask-mAP offset is removed.

The torch path (``materialize_masks_torch``) mirrors the same algorithm
entirely in PyTorch so mask materialization can run on the GPU when the
ONNX backend is using CUDAExecutionProvider. Inputs and outputs are NumPy
arrays (ORT always returns host numpy; the final masks must be numpy for
pycocotools); the function pushes tensors to the requested device, does all
heavy ops on-device, and copies back only the final binary uint8 masks.
This keeps the torch path device-local (no per-pixel host round-trips
mid-computation) while leaving the numpy path unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np

from .letterbox import LetterboxInfo


def materialize_masks_numpy(protos, coeffs, boxes_lb, lb: LetterboxInfo,
                            input_w: int, input_h: int) -> list[np.ndarray]:
    """Materialize per-detection binary masks at original image resolution.

    Args:
        protos: (1, c, mh, mw) mask prototypes.
        coeffs: (N, c) mask coefficients (post-NMS).
        boxes_lb: (N, 4) xyxy boxes in **letterboxed input** pixel space
            (pre-unletterbox). Used for the input-resolution crop, matching
            Ultralytics ``crop_mask``.
        lb: letterbox geometry (for the input -> original pad strip).
        input_w, input_h: model input size (square for Ultralytics export).
    """
    if coeffs is None or len(coeffs) == 0 or boxes_lb is None or len(boxes_lb) == 0:
        return []
    c, mh, mw = protos.shape[1], protos.shape[2], protos.shape[3]
    proto = protos[0].astype(np.float32).reshape(c, -1)            # (c, mh*mw)
    logits = (coeffs.astype(np.float32) @ proto).reshape(-1, mh, mw)  # (N, mh, mw) raw

    # input -> original pad strip (scale_masks semantics: round(pad -/+ 0.1)).
    left = int(round(lb.pad_x - 0.1))
    top = int(round(lb.pad_y - 0.1))
    right = input_w - int(round(lb.pad_x + 0.1))
    bottom = input_h - int(round(lb.pad_y + 0.1))

    # column / row pixel indices for the input-resolution crop_mask.
    r = np.arange(input_w, dtype=np.float32)   # x (cols)
    col_c = np.arange(input_h, dtype=np.float32)  # y (rows)

    out: list[np.ndarray] = []
    for m, box in zip(logits, np.asarray(boxes_lb, dtype=np.float32)):
        # scale_masks: proto resolution -> input resolution (square, no pad crop).
        m_in = cv2.resize(m, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        # crop_mask at input resolution: zero logits outside the bbox (float bounds).
        x1, y1, x2, y2 = box
        col_keep = (r >= x1) & (r < x2)          # (input_w,)
        row_keep = (col_c >= y1) & (col_c < y2)  # (input_h,)
        m_in = m_in * row_keep[:, None] * col_keep[None, :]
        # threshold raw logits > 0 -> binary at input resolution.
        bin_in = (m_in > 0.0).astype(np.float32)
        # scale_masks: input -> original (strip letterbox pad, resize), then floor.
        crop = bin_in[top:bottom, left:right]
        if crop.size == 0:
            out.append(np.zeros((lb.orig_h, lb.orig_w), np.uint8))
            continue
        m_ori = cv2.resize(crop, (lb.orig_w, lb.orig_h), interpolation=cv2.INTER_LINEAR)
        out.append(m_ori.astype(np.uint8))   # floor (truncate) == torch .byte()
    return out


def materialize_masks_torch(protos, coeffs, boxes_lb, lb: LetterboxInfo,
                            input_w: int, input_h: int,
                            device: str = "cpu") -> "torch.Tensor":
    """Materialize per-detection binary masks using PyTorch ops.

    Implements the same algorithm as ``materialize_masks_numpy`` but entirely
    in PyTorch so that when ``device="cuda"`` all heavy tensor ops (matmul,
    resize, crop) run on-GPU. Inputs are NumPy arrays (ORT always returns host
    memory regardless of CUDAExecutionProvider); the output tensor stays
    on-device — no D2H transfer happens here. The caller is responsible for
    the single batched D2H copy and RLE encoding (see ``detections_to_coco``).

    The algorithm is identical to the numpy path:
        logits = coeffs @ proto.reshape(c, mh*mw)   -> (N, mh, mw)
        -> upsample bilinear -> (N, input_h, input_w)
        -> zero cols/rows outside bbox (float bounds)
        -> threshold > 0 -> binary
        -> strip letterbox pad, downsample bilinear -> (N, orig_h, orig_w)
        -> floor (.byte())

    Args:
        protos: (1, c, mh, mw) mask prototypes — numpy array from ORT output.
        coeffs: (N, c) mask coefficients — numpy array from postprocessor.
        boxes_lb: (N, 4) xyxy boxes in letterboxed input pixel space.
        lb: letterbox geometry.
        input_w, input_h: model input size.
        device: torch device string ("cpu" or "cuda").

    Returns:
        torch.Tensor of shape (N, orig_h, orig_w), dtype=uint8, on ``device``.
        Returns an empty list ``[]`` when there are no detections.
    """
    import torch
    import torch.nn.functional as F

    if coeffs is None or len(coeffs) == 0 or boxes_lb is None or len(boxes_lb) == 0:
        return []

    c, mh, mw = int(protos.shape[1]), int(protos.shape[2]), int(protos.shape[3])
    N = len(coeffs)

    # Push inputs to device once.
    proto_t = torch.from_numpy(np.asarray(protos[0], dtype=np.float32)).to(device)   # (c, mh, mw)
    coeff_t = torch.from_numpy(np.asarray(coeffs, dtype=np.float32)).to(device)      # (N, c)
    boxes_t = torch.from_numpy(np.asarray(boxes_lb, dtype=np.float32)).to(device)    # (N, 4)

    # logits: (N, mh*mw) -> (N, mh, mw)
    logits = (coeff_t @ proto_t.reshape(c, -1)).reshape(N, mh, mw)  # (N, mh, mw)

    # scale_masks: proto resolution -> input resolution via bilinear upsample.
    # F.interpolate expects (N, C, H, W); treat each mask as a 1-channel map.
    logits_in = F.interpolate(
        logits.unsqueeze(1),                       # (N, 1, mh, mw)
        size=(input_h, input_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)                                   # (N, input_h, input_w)

    # crop_mask at input resolution: zero logits outside each bbox (float bounds).
    # Build coordinate grids on-device.
    col_idx = torch.arange(input_w, dtype=torch.float32, device=device)           # (input_w,) -> (N,1,W)
    row_idx = torch.arange(input_h, dtype=torch.float32, device=device).view(-1, 1)  # (input_h,1) -> (N,H,1)

    x1 = boxes_t[:, 0].view(N, 1, 1)   # (N,1,1)
    y1 = boxes_t[:, 1].view(N, 1, 1)
    x2 = boxes_t[:, 2].view(N, 1, 1)
    y2 = boxes_t[:, 3].view(N, 1, 1)

    col_keep = ((col_idx >= x1) & (col_idx < x2)).float()   # (N, 1, input_w)
    row_keep = ((row_idx >= y1) & (row_idx < y2)).float()   # (N, input_h, 1)
    logits_in = logits_in * row_keep * col_keep              # (N, input_h, input_w)

    # Threshold > 0 -> binary float.
    bin_in = (logits_in > 0.0).float()                       # (N, input_h, input_w)

    # Letterbox pad boundaries (scale_masks: round(pad -/+ 0.1)).
    left   = int(round(lb.pad_x - 0.1))
    top    = int(round(lb.pad_y - 0.1))
    right  = input_w - int(round(lb.pad_x + 0.1))
    bottom = input_h - int(round(lb.pad_y + 0.1))

    # Strip letterbox padding then downsample to original resolution.
    crop = bin_in[:, top:bottom, left:right]                 # (N, crop_h, crop_w)
    if crop.shape[1] == 0 or crop.shape[2] == 0:
        return torch.zeros((N, lb.orig_h, lb.orig_w), dtype=torch.uint8, device=device)

    m_ori = F.interpolate(
        crop.unsqueeze(1),                         # (N, 1, crop_h, crop_w)
        size=(lb.orig_h, lb.orig_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)                                   # (N, orig_h, orig_w)

    # floor (truncate) -> binary uint8, matching torch .byte() semantics.
    # Stay on device — caller (coco_output) does the single batched D2H.
    return m_ori.to(torch.uint8)                   # (N, orig_h, orig_w) uint8, on device
