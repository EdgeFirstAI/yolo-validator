# tests/test_masks.py
import numpy as np
import pytest
from yolo_validator.masks import materialize_masks_numpy, materialize_masks_torch
from yolo_validator.letterbox import compute_letterbox


def test_materialize_masks_shapes_and_binary():
    # one proto channel fully active; coeff selects it -> mask covers its bbox
    protos = np.zeros((1, 32, 160, 160), dtype=np.float32)
    protos[0, 0] = 10.0   # strong positive logit everywhere
    coeffs = np.zeros((1, 32), dtype=np.float32)
    coeffs[0, 0] = 1.0
    boxes = np.array([[100.0, 100.0, 300.0, 300.0]], dtype=np.float32)  # orig coords
    lb = compute_letterbox(640, 480, 640, 640, scaleup=False)
    masks = materialize_masks_numpy(protos, coeffs, boxes, lb, 640, 640)
    assert len(masks) == 1
    m = masks[0]
    assert m.shape == (480, 640)            # original image size
    assert set(np.unique(m)).issubset({0, 1})
    # inside the bbox the mask is on; outside it is off
    assert m[200, 200] == 1
    assert m[10, 10] == 0


def test_materialize_masks_empty():
    protos = np.zeros((1, 32, 160, 160), dtype=np.float32)
    out = materialize_masks_numpy(protos, np.zeros((0, 32), np.float32),
                                  np.zeros((0, 4), np.float32),
                                  compute_letterbox(640, 640, 640, 640), 640, 640)
    assert out == []


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_masks_match_numpy(device):
    """torch path (CPU and CUDA) must be pixel-identical to the numpy path."""
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    rng = np.random.default_rng(42)
    c, mh, mw = 32, 160, 160
    input_w, input_h = 640, 640
    orig_w, orig_h = 640, 480
    lb = compute_letterbox(orig_w, orig_h, input_w, input_h)

    protos = rng.standard_normal((1, c, mh, mw)).astype(np.float32)
    coeffs = rng.standard_normal((8, c)).astype(np.float32)
    x1 = rng.uniform(50, 450, 8)
    y1 = rng.uniform(lb.pad_y + 10, input_h - lb.pad_y - 60, 8)
    boxes_lb = np.stack([x1, y1, x1 + rng.uniform(40, 120, 8),
                         y1 + rng.uniform(40, 120, 8)], axis=1).astype(np.float32)

    ref = materialize_masks_numpy(protos, coeffs, boxes_lb, lb, input_w, input_h)
    got_t = materialize_masks_torch(protos, coeffs, boxes_lb, lb, input_w, input_h, device=device)
    got = list(got_t.cpu().numpy())   # deferred D2H — same pattern as production

    assert len(got) == len(ref)
    for i, (r, g) in enumerate(zip(ref, got)):
        assert r.shape == g.shape, f"mask {i} shape mismatch: {r.shape} vs {g.shape}"
        diff = int(np.sum(r != g))
        assert diff == 0, f"mask {i} pixel diff = {diff} (device={device})"


def test_torch_masks_bbox_crop_uses_correct_axis():
    """Regression: row_idx must broadcast as (H,1) not (1,H) so y-bounds apply to rows."""
    torch = pytest.importorskip("torch")
    # Asymmetric bbox: only the y-range [y1=300, y2=400] should zero rows
    # outside 300-400; columns are full-width. If row_idx is wrong axis,
    # the y-bounds silently become column-bounds and the mask is wrong.
    protos = np.zeros((1, 32, 160, 160), dtype=np.float32)
    protos[0, 0] = 10.0   # strong positive everywhere
    coeffs = np.zeros((1, 32), dtype=np.float32)
    coeffs[0, 0] = 1.0
    boxes_lb = np.array([[0.0, 300.0, 640.0, 400.0]], dtype=np.float32)
    lb = compute_letterbox(640, 640, 640, 640)

    ref = materialize_masks_numpy(protos, coeffs, boxes_lb, lb, 640, 640)
    got = list(materialize_masks_torch(protos, coeffs, boxes_lb, lb, 640, 640, device="cpu").numpy())

    # The mask should be nonzero only in rows 300-400.
    r, g = ref[0], got[0]
    assert r.shape == (640, 640)
    assert g.shape == (640, 640)
    diff = int(np.sum(r != g))
    assert diff == 0, f"Axis-swap regression: torch mask differs from numpy by {diff} pixels"
    # Sanity: rows outside the bbox should be zero
    assert r[:300].sum() == 0, "rows above bbox should be zero"
    assert r[400:].sum() == 0, "rows below bbox should be zero"
