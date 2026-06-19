# tests/test_coco_eval.py
import json
import numpy as np
from pycocotools import mask as mask_utils
from yolo_validator.coco_eval import evaluate_coco


def _tiny_gt(tmp_path):
    # one image, one GT box (and segm) of category 1
    m = np.zeros((20, 20), dtype=np.uint8)
    m[5:15, 5:15] = 1
    rle = mask_utils.encode(np.asfortranarray(m))
    rle["counts"] = rle["counts"].decode("ascii")
    gt = {
        "images": [{"id": 1, "width": 20, "height": 20}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 1,
            "bbox": [5, 5, 10, 10], "area": 100, "iscrowd": 0,
            "segmentation": rle,
        }],
    }
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(gt))
    return str(p), rle


def test_evaluate_coco_perfect_bbox(tmp_path):
    gt_path, rle = _tiny_gt(tmp_path)
    preds = [{"image_id": 1, "category_id": 1, "bbox": [5, 5, 10, 10],
              "score": 0.9, "segmentation": rle}]
    res = evaluate_coco(gt_path, preds, iou_types=("bbox", "segm"))
    assert res["bbox"]["AP"] > 0.99   # perfect overlap
    assert res["segm"]["AP"] > 0.99
