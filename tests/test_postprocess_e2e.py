# tests/test_postprocess_e2e.py
"""Tests for E2EPostprocessor."""
import numpy as np
import pytest
from yolo_validator.postprocess_e2e import E2EPostprocessor
from yolo_validator.letterbox import compute_letterbox, LetterboxInfo
from yolo_validator.backends import ModelSpec


def _make_lb():
    return compute_letterbox(640, 640, 640, 640)


def test_e2e_detect_basic():
    """E2E detect: [1,3,6] with 2 rows above conf, 1 below → 2 detections."""
    spec = ModelSpec(640, 640, "detect", e2e=True)
    lb = _make_lb()
    # x1, y1, x2, y2, score, class
    rows = np.array([
        [100.0, 100.0, 200.0, 200.0, 0.9, 3.0],
        [300.0, 300.0, 400.0, 400.0, 0.8, 7.0],
        [  0.0,   0.0, 100.0, 100.0, 0.0001, 0.0],  # below conf
    ], dtype=np.float32)
    pred = rows[np.newaxis]  # (1, 3, 6)
    post = E2EPostprocessor(spec, conf=0.001, iou=0.7, max_det=300)
    det = post.decode([pred], lb)
    assert len(det.scores) == 2
    assert det.classes[0] == 3
    assert det.classes[1] == 7
    assert abs(det.scores[0] - 0.9) < 1e-5
    assert det.coeffs is None
    assert det.protos is None


def test_e2e_detect_empty():
    """E2E detect: all rows below conf → empty result."""
    spec = ModelSpec(640, 640, "detect", e2e=True)
    lb = _make_lb()
    rows = np.zeros((1, 3, 6), dtype=np.float32)
    post = E2EPostprocessor(spec, conf=0.5, iou=0.7, max_det=300)
    det = post.decode([rows], lb)
    assert len(det.scores) == 0
    assert det.boxes.shape == (0, 4)
    assert det.coeffs is None


def test_e2e_segment_coeffs():
    """E2E segment: [1,2,38] → coeffs shape (k,32), protos passed through."""
    spec = ModelSpec(640, 640, "segment", e2e=True)
    lb = _make_lb()
    nm = 32
    # 2 rows above conf, columns: x1,y1,x2,y2,score,class,32 coeffs
    rows = np.zeros((1, 2, 6 + nm), dtype=np.float32)
    rows[0, 0, 4] = 0.9  # score
    rows[0, 0, 5] = 1.0  # class
    rows[0, 1, 4] = 0.8
    rows[0, 1, 5] = 2.0
    protos = np.zeros((1, 32, 160, 160), dtype=np.float32)
    post = E2EPostprocessor(spec, conf=0.001, iou=0.7, max_det=300)
    det = post.decode([rows, protos], lb)
    assert len(det.scores) == 2
    assert det.coeffs is not None
    assert det.coeffs.shape == (2, 32)
    assert det.protos is protos


def test_e2e_max_det():
    """E2E: max_det limits output."""
    spec = ModelSpec(640, 640, "detect", e2e=True)
    lb = _make_lb()
    n = 10
    rows = np.zeros((1, n, 6), dtype=np.float32)
    rows[0, :, 4] = np.arange(n, 0, -1, dtype=np.float32) * 0.1  # scores
    post = E2EPostprocessor(spec, conf=0.001, iou=0.7, max_det=3)
    det = post.decode([rows], lb)
    assert len(det.scores) == 3
    # Should be top 3 by score (descending)
    assert det.scores[0] >= det.scores[1] >= det.scores[2]
