import numpy as np
from yolo_validator.letterbox import compute_letterbox, unletterbox_boxes


def test_compute_letterbox_landscape_no_upscale():
    # 1280x720 into 640x640, scaleup disabled (Ultralytics val default)
    lb = compute_letterbox(1280, 720, 640, 640, scaleup=False)
    assert lb.scale == 0.5
    assert lb.pad_x == 0.0
    assert lb.pad_y == (640 - 360) / 2.0  # 140
    assert lb.orig_w == 1280 and lb.orig_h == 720


def test_compute_letterbox_no_upscale_small_image():
    lb = compute_letterbox(320, 240, 640, 640, scaleup=False)
    assert lb.scale == 1.0  # capped, no upscaling


def test_unletterbox_roundtrip():
    lb = compute_letterbox(1280, 720, 640, 640, scaleup=False)
    # a box covering the whole original maps to letterboxed coords and back
    orig = np.array([[0.0, 0.0, 1280.0, 720.0]], dtype=np.float32)
    lboxed = orig.copy()
    lboxed[:, [0, 2]] = orig[:, [0, 2]] * lb.scale + lb.pad_x
    lboxed[:, [1, 3]] = orig[:, [1, 3]] * lb.scale + lb.pad_y
    back = unletterbox_boxes(lboxed, lb)
    np.testing.assert_allclose(back, orig, atol=1e-4)


def test_unletterbox_empty():
    lb = compute_letterbox(640, 480, 640, 640)
    out = unletterbox_boxes(np.zeros((0, 4), dtype=np.float32), lb)
    assert out.shape == (0, 4)
