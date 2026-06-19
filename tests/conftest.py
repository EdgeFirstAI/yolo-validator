import numpy as np
import cv2
import pytest


@pytest.fixture
def rgb_image(tmp_path):
    """Write a 1280x720 BGR test image to disk; return (path, w, h)."""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:, :, 2] = 200  # red-ish in BGR
    p = tmp_path / "img.jpg"
    cv2.imwrite(str(p), img)
    return str(p), 1280, 720


class MockBackend:
    """Backend stub producing a fixed YOLO-seg raw output for pipeline tests."""

    def __init__(self, task="segment", input_w=640, input_h=640, nc=80):
        from yolo_validator.backends import ModelSpec
        self.spec = ModelSpec(input_w=input_w, input_h=input_h, task=task)
        self._nc = nc

    def run(self, input_tensor):
        nm = 32 if self.spec.task == "segment" else 0
        ch = 4 + self._nc + nm
        pred = np.zeros((1, ch, 8400), dtype=np.float32)
        # one strong detection of class 0 at center of the input grid
        pred[0, 0, 0] = 320.0  # cx
        pred[0, 1, 0] = 320.0  # cy
        pred[0, 2, 0] = 200.0  # w
        pred[0, 3, 0] = 200.0  # h
        pred[0, 4, 0] = 0.95   # class 0 score
        outs = [pred]
        if nm:
            protos = np.zeros((1, 32, 160, 160), dtype=np.float32)
            protos[0, 0] = 5.0          # large positive logit channel
            pred[0, 4 + self._nc + 0, 0] = 1.0   # coeff selects channel 0
            outs.append(protos)
        return outs, None


@pytest.fixture
def mock_backend():
    return MockBackend()
