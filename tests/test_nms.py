import numpy as np
from yolo_validator.nms import nms, nms_class_aware


def test_nms_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, iou_thres=0.5)
    assert keep.tolist() == [0, 2]  # box 1 suppressed by box 0; box 2 is far away


def test_nms_class_aware_keeps_overlapping_diff_classes():
    # heavily overlapping boxes but different classes -> both survive
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 1], dtype=np.int64)
    keep = nms_class_aware(boxes, scores, classes, iou_thres=0.5)
    assert sorted(keep.tolist()) == [0, 1]


def test_nms_empty():
    keep = nms(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), 0.5)
    assert keep.shape == (0,)
