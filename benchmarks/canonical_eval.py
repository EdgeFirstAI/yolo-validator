# benchmarks/canonical_eval.py
"""Canonical COCOeval restricted to the image_ids present in predictions.

Re-scores predictions through one canonical pycocotools call, restricting
COCOeval.params.imgIds to the image_ids present in predictions. This ensures
subset runs and all configs are compared on the same image set (not the full
5000-image GT), so partial-run comparisons are apples-to-apples.
"""
from __future__ import annotations

import contextlib
import io
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl",
          "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]


def canonical_eval(
    gt_json_path: str,
    predictions: list[dict],
    iou_types: tuple[str, ...] = ("bbox",),
) -> dict:
    """Like evaluate_coco but restricts COCOeval.params.imgIds to the image_ids
    present in predictions. This ensures subset runs and all configs are compared
    on the same image set (not the full 5000-image GT).

    Args:
        gt_json_path: path to COCO ground-truth JSON (instances_val2017.json or subset).
        predictions: list of COCO-format prediction dicts (each must have 'image_id').
        iou_types: tuple of iou_type strings, e.g. ("bbox",) or ("bbox", "segm").

    Returns:
        Same dict structure as evaluate_coco: {iou_type: {metric_name: float}}
        where metric names are AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl.
    """
    if not predictions:
        return {t: {n: 0.0 for n in _NAMES} for t in iou_types}

    with contextlib.redirect_stdout(io.StringIO()):
        # iscrowd filtering is not currently supported: every ground-truth
        # annotation is scored as a normal target (crowd annotations are NOT
        # ignored), matching how the EdgeFirst stack validates.
        with open(gt_json_path, encoding="utf-8") as f:
            gt = json.load(f)
        for ann in gt.get("annotations", []):
            ann["iscrowd"] = 0
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        img_ids = list(set(p["image_id"] for p in predictions))
        coco_dt = coco_gt.loadRes(predictions)
        results: dict[str, dict] = {}
        for t in iou_types:
            ev = COCOeval(coco_gt, coco_dt, iouType=t)
            ev.params.imgIds = img_ids  # restrict to images we actually ran
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            results[t] = {n: float(v) for n, v in zip(_NAMES, ev.stats)}
    return results
