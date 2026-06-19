# tests/test_postprocess.py
import numpy as np
import pytest
from yolo_validator.postprocess import make_postprocessor
from yolo_validator.letterbox import compute_letterbox
from yolo_validator.backends import ModelSpec


def _seg_raw(nc=80):
    pred = np.zeros((1, 4 + nc + 32, 8400), dtype=np.float32)
    pred[0, 0, 0] = 320.0
    pred[0, 1, 0] = 320.0
    pred[0, 2, 0] = 200.0
    pred[0, 3, 0] = 200.0
    pred[0, 4, 0] = 0.95
    protos = np.zeros((1, 32, 160, 160), dtype=np.float32)
    return [pred, protos]


def test_numpy_decode_one_detection():
    spec = ModelSpec(640, 640, "segment")
    lb = compute_letterbox(1280, 720, 640, 640, scaleup=False)
    post = make_postprocessor("numpy", spec, conf=0.001, iou=0.7, max_det=300)
    det = post.decode(_seg_raw(), lb)
    assert len(det.scores) == 1
    assert det.classes[0] == 0
    assert det.scores[0] == np.float32(0.95)
    # center box 320,320 in letterboxed space -> original coords (un-padded, /scale)
    # pad_y=140, scale=0.5  => cy_orig = (320-140)/0.5 = 360
    x0, y0, x1, y1 = det.boxes[0]
    assert abs(((y0 + y1) / 2) - 360.0) < 2.0
    assert det.coeffs is not None and det.protos is not None


def test_numpy_decode_detect_only_has_no_masks():
    spec = ModelSpec(640, 640, "detect")
    lb = compute_letterbox(640, 640, 640, 640)
    pred = np.zeros((1, 84, 8400), dtype=np.float32)
    pred[0, 2, 0] = 50; pred[0, 3, 0] = 50; pred[0, 4, 0] = 0.9
    post = make_postprocessor("numpy", spec, conf=0.001, iou=0.7, max_det=300)
    det = post.decode([pred], lb)
    assert det.coeffs is None and det.protos is None
    assert len(det.scores) == 1


def test_torch_decode_matches_numpy_boxes():
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from yolo_validator.postprocess import NumpyPostprocessor
    from yolo_validator.postprocess_torch import TorchPostprocessor

    spec = ModelSpec(640, 640, "segment")
    lb = compute_letterbox(1280, 720, 640, 640, scaleup=False)
    raw = _seg_raw()
    d_np = NumpyPostprocessor(spec, 0.001, 0.7, 300).decode(raw, lb)
    d_t = TorchPostprocessor(spec, 0.001, 0.7, 300).decode(raw, lb)
    assert len(d_t.scores) == len(d_np.scores) == 1
    np.testing.assert_allclose(np.sort(d_t.boxes.ravel()),
                               np.sort(d_np.boxes.ravel()), atol=2.0)


def test_numpy_multilabel_two_classes():
    """Multi-label: one anchor with two classes above conf → 2 detections."""
    nc = 2
    spec = ModelSpec(640, 640, "detect")
    lb = compute_letterbox(640, 640, 640, 640)
    pred = np.zeros((1, 4 + nc, 8400), dtype=np.float32)
    # Anchor 0: box at center 320,320 size 100,100; both classes above conf
    pred[0, 0, 0] = 320.0  # cx
    pred[0, 1, 0] = 320.0  # cy
    pred[0, 2, 0] = 100.0  # w
    pred[0, 3, 0] = 100.0  # h
    pred[0, 4, 0] = 0.8    # class 0 score
    pred[0, 5, 0] = 0.7    # class 1 score
    from yolo_validator.postprocess import NumpyPostprocessor
    post = NumpyPostprocessor(spec, conf=0.001, iou=0.7, max_det=300)
    det = post.decode([pred], lb)
    assert len(det.scores) == 2, f"expected 2 detections, got {len(det.scores)}"
    assert set(det.classes.tolist()) == {0, 1}


def test_torch_multilabel_two_classes():
    """Torch multi-label: one anchor with two classes above conf → 2 detections."""
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    nc = 2
    spec = ModelSpec(640, 640, "detect")
    lb = compute_letterbox(640, 640, 640, 640)
    pred = np.zeros((1, 4 + nc, 8400), dtype=np.float32)
    pred[0, 0, 0] = 320.0
    pred[0, 1, 0] = 320.0
    pred[0, 2, 0] = 100.0
    pred[0, 3, 0] = 100.0
    pred[0, 4, 0] = 0.8
    pred[0, 5, 0] = 0.7
    from yolo_validator.postprocess_torch import TorchPostprocessor
    post = TorchPostprocessor(spec, conf=0.001, iou=0.7, max_det=300)
    det = post.decode([pred], lb)
    assert len(det.scores) == 2, f"expected 2 detections, got {len(det.scores)}"
    assert set(det.classes.tolist()) == {0, 1}
