"""Pure-NumPy unit tests for the Hailo on-device decoders.

The Hailo decode paths (`_dfl`, `_decode_nmsfree`, `_decode_seg`) are otherwise
only exercised on the Pi against real HEF outputs. These tests feed synthetic
tensors with a single planted detection so shape/stride/decode regressions are
caught in CI without Hailo hardware.

Decode geometry under test (imgsz=640, one stride grid h=20 -> stride=32):
anchor 0 sits at cell-center (0.5, 0.5); a distance box (l,t,r,b) decodes to
xyxy = ((0.5-l)*32, (0.5-t)*32, (0.5+r)*32, (0.5+b)*32).
"""
import numpy as np
import pytest

from benchmarks.hailo_infer import (_decode_nmsfree, _decode_seg, _dfl,
                                    _sigmoid)
from yolo_validator.coco_output import coco80_to_coco91
from yolo_validator.letterbox import LetterboxInfo


def test_dfl_integrates_bins():
    """DFL softmax-integrates each 16-bin group to its argmax index."""
    dists = [3, 5, 2, 7]
    box = np.full((1, 64), -100.0, np.float32)
    for g, k in enumerate(dists):
        box[0, g * 16 + k] = 100.0  # ~one-hot -> integrated distance == k
    out = _dfl(box)
    assert out.shape == (1, 4)
    np.testing.assert_allclose(out[0], dists, atol=1e-3)


def _nmsfree_results(stride_h=20, nc=80):
    """One-stride NMS-free head with a single planted detection at anchor 0."""
    box = np.zeros((1, stride_h, stride_h, 4), np.float32)
    box[0, 0, 0] = [0.0, 0.0, 2.0, 2.0]          # l,t,r,b -> xyxy [16,16,80,80]
    cls = np.full((1, stride_h, stride_h, nc), -10.0, np.float32)
    cls[0, 0, 0, 0] = 10.0                        # class 0 is the only one > thr
    return {"box": box, "cls": cls}


def test_decode_nmsfree_single_box():
    recs = _decode_nmsfree(_nmsfree_results(), imgsz=640, scale=1.0, pad_x=0,
                           pad_y=0, w0=640, h0=640, class_map=coco80_to_coco91(),
                           image_id=42, score_th=0.001)
    assert len(recs) == 1
    r = recs[0]
    assert r["image_id"] == 42
    assert r["category_id"] == coco80_to_coco91()[0]
    np.testing.assert_allclose(r["bbox"], [16, 16, 64, 64], atol=1e-3)
    assert r["score"] == pytest.approx(float(_sigmoid(np.float32(10.0))), abs=1e-5)


def test_decode_nmsfree_empty_below_threshold():
    """No class logit above threshold -> no detections (no crash)."""
    cls = np.full((1, 20, 20, 80), -10.0, np.float32)
    res = {"box": np.zeros((1, 20, 20, 4), np.float32), "cls": cls}
    assert _decode_nmsfree(res, 640, 1.0, 0, 0, 640, 640,
                           coco80_to_coco91(), 1, 0.001) == []


def _seg_results(stride_h=20, nc=80, nm=32, proto_h=160):
    """One-stride yolov8-seg head + prototype, one planted detection."""
    # DFL box: distances l=0,t=0,r=4,b=4 -> xyxy [16,16,144,144] at anchor 0.
    box = np.full((1, stride_h, stride_h, 64), -10.0, np.float32)
    for g, k in zip(range(4), (0, 0, 4, 4)):
        box[0, 0, 0, g * 16 + k] = 10.0
    cls = np.full((1, stride_h, stride_h, nc), -10.0, np.float32)
    cls[0, 0, 0, 0] = 10.0
    coeff = np.zeros((1, stride_h, stride_h, nm), np.float32)
    coeff[0, 0, 0] = 0.5
    proto = np.zeros((1, proto_h, proto_h, nm), np.float32)
    proto[0, :, :, 0] = 4.0  # positive activation so the mask is non-empty
    return {"box": box, "cls": cls, "coeff": coeff, "proto": proto}


def test_decode_seg_single_instance():
    lb = LetterboxInfo(1.0, 0.0, 0.0, 640, 640)
    det, masks = _decode_seg(_seg_results(), imgsz=640, lb=lb, score_th=0.001)
    assert len(det.scores) == 1
    assert det.classes[0] == 0
    np.testing.assert_allclose(det.boxes[0], [16, 16, 144, 144], atol=1.0)
    assert len(masks) == 1
    assert masks[0].shape == (640, 640)
    assert masks[0].dtype == np.uint8


def test_decode_seg_empty_below_threshold():
    """No class logit above threshold -> empty Detections + no masks."""
    res = _seg_results()
    res["cls"][:] = -10.0
    lb = LetterboxInfo(1.0, 0.0, 0.0, 640, 640)
    det, masks = _decode_seg(res, 640, lb, 0.001)
    assert len(det.scores) == 0
    assert masks == []


def test_decode_seg_missing_proto_raises():
    """A segment head with no prototype tensor fails loudly, not with an
    obscure AttributeError deep in the decode."""
    res = _seg_results()
    del res["proto"]
    lb = LetterboxInfo(1.0, 0.0, 0.0, 640, 640)
    with pytest.raises(ValueError, match="prototype"):
        _decode_seg(res, 640, lb, 0.001)
