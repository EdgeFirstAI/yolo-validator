from yolo_validator.imageio import decode_image


def test_decode_image_returns_bgr_and_size(rgb_image):
    path, w, h = rgb_image
    img, gw, gh = decode_image(path)
    assert (gw, gh) == (w, h)
    assert img.shape == (h, w, 3)
    assert img.dtype.name == "uint8"


import numpy as np
from yolo_validator.preprocess import make_preprocessor


def test_numpy_preprocess_shape_and_range(rgb_image):
    path, w, h = rgb_image
    img, gw, gh = decode_image(path)
    pre = make_preprocessor("numpy", input_w=640, input_h=640)
    tensor, lb = pre.preprocess(img, gw, gh)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0
    assert lb.orig_w == w and lb.orig_h == h
    assert lb.scale == 0.5  # 1280x720 -> 640


def test_numpy_preprocess_is_rgb(rgb_image):
    # input is red in BGR; after BGR->RGB the red should land in channel 0
    path, w, h = rgb_image
    img, gw, gh = decode_image(path)
    tensor, lb = make_preprocessor("numpy", input_w=640, input_h=640).preprocess(img, gw, gh)
    # center pixel is image content (not pad)
    cx, cy = 320, 320
    assert tensor[0, 0, cy, cx] > tensor[0, 2, cy, cx]  # R > B


import pytest


def test_torch_preprocess_matches_numpy(rgb_image):
    pytest.importorskip("torch")  # skip ONLY this test without torch (keep numpy coverage)
    from yolo_validator.preprocess import NumpyPreprocessor
    from yolo_validator.preprocess_torch import TorchPreprocessor
    from yolo_validator.imageio import decode_image

    path, w, h = rgb_image
    img, gw, gh = decode_image(path)
    t_np, lb_np = NumpyPreprocessor(640, 640).preprocess(img, gw, gh)
    t_t, lb_t = TorchPreprocessor(640, 640).preprocess(img, gw, gh)
    assert t_t.shape == t_np.shape
    assert lb_t.scale == lb_np.scale
    # tensors agree closely (both are letterbox + /255 RGB)
    np.testing.assert_allclose(t_t, t_np, atol=2e-2)
