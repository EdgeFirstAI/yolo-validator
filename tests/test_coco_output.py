# tests/test_coco_output.py
import numpy as np
from yolo_validator.coco_output import detections_to_coco, coco80_to_coco91
from yolo_validator.detections import Detections


def test_detections_to_coco_bbox_xywh():
    det = Detections(
        boxes=np.array([[10.0, 20.0, 30.0, 60.0]], dtype=np.float32),
        scores=np.array([0.8], dtype=np.float32),
        classes=np.array([0], dtype=np.int64),
    )
    recs = detections_to_coco(image_id=42, det=det, class_map=coco80_to_coco91(), masks=None)
    assert len(recs) == 1
    r = recs[0]
    assert r["image_id"] == 42
    assert r["category_id"] == 1            # coco80 idx 0 -> coco91 id 1
    assert r["bbox"] == [10.0, 20.0, 20.0, 40.0]  # xywh
    assert abs(r["score"] - 0.8) < 1e-5
    assert "segmentation" not in r


def test_detections_to_coco_with_rle_mask():
    det = Detections(
        boxes=np.array([[0.0, 0.0, 5.0, 5.0]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        classes=np.array([0], dtype=np.int64),
    )
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:5, 0:5] = 1
    recs = detections_to_coco(42, det, coco80_to_coco91(), masks=[mask])
    assert recs[0]["segmentation"]["counts"]  # RLE string present
    assert isinstance(recs[0]["segmentation"]["counts"], str)
