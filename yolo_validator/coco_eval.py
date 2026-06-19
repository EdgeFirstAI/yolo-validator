# yolo_validator/coco_eval.py
"""pycocotools COCOeval wrapper (canonical evaluator).

Returns the 12 standard COCO metrics per iou_type. This is the single
canonical eval the comparison harness will later reuse to re-score every
side's predictions, so eval-library differences never masquerade as
prediction differences.
"""
from __future__ import annotations

import contextlib
import io

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl",
          "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]


def evaluate_coco(gt_json_path: str, predictions: list[dict], iou_types=("bbox",)) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(gt_json_path)
        results: dict[str, dict] = {}
        if not predictions:
            return {t: {n: 0.0 for n in _NAMES} for t in iou_types}
        coco_dt = coco_gt.loadRes(predictions)
        for t in iou_types:
            ev = COCOeval(coco_gt, coco_dt, iouType=t)
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            results[t] = {n: float(v) for n, v in zip(_NAMES, ev.stats)}
    return results
