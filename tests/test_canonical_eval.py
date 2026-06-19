# tests/test_canonical_eval.py
"""TDD tests for benchmarks/canonical_eval.py.

Verifies that canonical_eval restricts COCOeval to only the image_ids in
predictions, so partial-run evaluations are scored on the right denominator.
"""
from __future__ import annotations

import json
import pytest
from benchmarks.canonical_eval import canonical_eval
from yolo_validator.coco_eval import evaluate_coco


@pytest.fixture
def tiny_gt_json(tmp_path):
    """Tiny COCO GT with 2 images and one annotation each.

    image 1: category 1 bbox [10, 10, 50, 50] (xywh)
    image 2: category 1 bbox [20, 20, 60, 60] (xywh)
    """
    gt = {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img2.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [10.0, 10.0, 50.0, 50.0], "area": 2500.0, "iscrowd": 0,
            },
            {
                "id": 2, "image_id": 2, "category_id": 1,
                "bbox": [20.0, 20.0, 60.0, 60.0], "area": 3600.0, "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "thing", "supercategory": "object"}],
    }
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(gt))
    return str(p)


@pytest.fixture
def perfect_pred_image1():
    """A near-perfect bbox prediction for image 1 only."""
    return [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [10.0, 10.0, 50.0, 50.0],
            "score": 0.99,
        }
    ]


def test_canonical_eval_perfect_single_image(tiny_gt_json, perfect_pred_image1):
    """canonical_eval with predictions only for image 1 → near-perfect AP."""
    results = canonical_eval(tiny_gt_json, perfect_pred_image1, iou_types=("bbox",))
    assert "bbox" in results
    ap = results["bbox"]["AP"]
    ap50 = results["bbox"]["AP50"]
    # Perfect prediction on the one evaluated image should give AP≈1.0
    assert ap > 0.99, f"Expected AP~1.0 for perfect prediction, got {ap}"
    assert ap50 > 0.99, f"Expected AP50~1.0 for perfect prediction, got {ap50}"


def test_canonical_eval_restricts_to_predicted_images(tiny_gt_json, perfect_pred_image1):
    """canonical_eval should restrict to image_ids in predictions; evaluate_coco doesn't.

    With predictions only for image 1:
    - canonical_eval: evaluates over {image 1} → AP~1.0 (perfect)
    - evaluate_coco:  evaluates over {image 1, image 2} → AP~0.5 (image 2 has no predictions)
    """
    canonical_results = canonical_eval(tiny_gt_json, perfect_pred_image1)
    old_results = evaluate_coco(tiny_gt_json, perfect_pred_image1)

    canonical_ap = canonical_results["bbox"]["AP"]
    old_ap = old_results["bbox"]["AP"]

    # canonical_eval should give near-perfect AP (restricted to image 1)
    assert canonical_ap > 0.99, f"canonical_eval AP={canonical_ap}, expected ~1.0"
    # evaluate_coco averages over both images → AP should be ~0.5
    assert old_ap < 0.6, f"evaluate_coco AP={old_ap}, expected ~0.5 (penalised by image 2)"
    # They must differ because of the imgIds restriction
    assert canonical_ap > old_ap + 0.3, (
        f"canonical_eval ({canonical_ap:.3f}) should be much higher than "
        f"evaluate_coco ({old_ap:.3f}) when only predicting one of two images"
    )


def test_canonical_eval_empty_predictions(tiny_gt_json):
    """Empty predictions → all zeros."""
    results = canonical_eval(tiny_gt_json, [], iou_types=("bbox",))
    assert results["bbox"]["AP"] == 0.0
    assert results["bbox"]["AP50"] == 0.0


def test_canonical_eval_both_images(tiny_gt_json):
    """Predictions for both images → AP should be high if both are near-perfect."""
    preds = [
        {"image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 50.0, 50.0], "score": 0.99},
        {"image_id": 2, "category_id": 1, "bbox": [20.0, 20.0, 60.0, 60.0], "score": 0.99},
    ]
    results = canonical_eval(tiny_gt_json, preds)
    assert results["bbox"]["AP"] > 0.99


def test_canonical_eval_multiple_iou_types(tmp_path):
    """canonical_eval accepts multiple iou_types, returns a key per type."""
    # Build a GT with a segmentation annotation
    gt = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [5.0, 5.0, 20.0, 20.0], "area": 400.0, "iscrowd": 0,
                "segmentation": [[5, 5, 25, 5, 25, 25, 5, 25]],
            }
        ],
        "categories": [{"id": 1, "name": "thing", "supercategory": "object"}],
    }
    gt_path = str(tmp_path / "gt_seg.json")
    with open(gt_path, "w") as f:
        json.dump(gt, f)

    # Provide a segmentation prediction too (RLE or polygon)
    preds = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [5.0, 5.0, 20.0, 20.0],
            "score": 0.9,
            "segmentation": {"size": [100, 100], "counts": "0"},  # minimal RLE placeholder
        }
    ]
    # Just verify both iou_types are returned (scores may be low due to placeholder seg)
    results = canonical_eval(gt_path, preds, iou_types=("bbox", "segm"))
    assert "bbox" in results
    assert "segm" in results
    assert set(results["bbox"].keys()) == {"AP", "AP50", "AP75", "APs", "APm", "APl",
                                            "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"}
