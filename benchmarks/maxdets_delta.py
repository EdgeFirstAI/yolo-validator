# benchmarks/maxdets_delta.py
"""Quantify the validator-ng maxDets 100->300 effect on box mAP.

validator-ng's metrics.py now overrides COCOeval.params.maxDets = [1, 10, 300]
(was the pycocotools default [1, 10, 100]). This measures the resulting box AP
delta on real COCO val2017 predictions, using the SAME engine validator-ng uses
(faster-coco-eval) and the SAME crowd handling EdgeFirst uses (crowd-as-normal,
iscrowd forced to 0). Predictions come from the square-letterbox (rect=False)
Ultralytics val, i.e. the geometry the EdgeFirst / yolo-validator proxy lanes use.

delta = AP@maxDets=300 - AP@maxDets=100  (all else identical).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

from benchmarks.runners import run_ultralytics

REPO = Path(__file__).resolve().parent.parent


def score(gt_json_path: str, predictions: list[dict], max_dets: list[int],
          iou_type: str = "bbox") -> float:
    """faster-coco-eval box AP at a given maxDets, crowd-as-normal, restricted to
    the predicted image_ids (mirrors canonical_eval)."""
    from faster_coco_eval import COCO, COCOeval_faster as COCOeval
    with open(gt_json_path, encoding="utf-8") as f:
        gt = json.load(f)
    for ann in gt.get("annotations", []):
        ann["iscrowd"] = 0
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(gt)
        img_ids = sorted({p["image_id"] for p in predictions})
        coco_dt = coco_gt.loadRes(predictions)
        ev = COCOeval(coco_gt, coco_dt, iou_type)
        ev.params.imgIds = img_ids
        ev.params.maxDets = list(max_dets)
        ev.evaluate(); ev.accumulate(); ev.summarize()
        return float(ev.stats[0])  # AP@0.5:0.95


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["yolov8n"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-yaml",
                   default=str(REPO / "benchmarks/_coco_shadow/coco-val.yaml"))
    p.add_argument("--gt-json",
                   default=str(Path("~/coco/annotations/instances_val2017.json").expanduser()))
    args = p.parse_args()
    device = 0 if args.device == "cuda" else args.device

    print("maxDets 100 vs 300 (box AP, crowd-as-normal, rect=False):\n")
    for model in args.models:
        task = "segment" if model.endswith("-seg") else "detect"
        pt = str(REPO / f"{model}.pt")
        from ultralytics import YOLO
        from ultralytics.nn.modules import Detect
        m = YOLO(pt)
        for mod in m.model.modules():
            if isinstance(mod, Detect):
                mod.end2end = False
                break
        res = run_ultralytics(pt, args.data_yaml, task, pre_val_model=m,
                              device=device, rect=False)
        preds = res["predictions"]
        ap100 = score(args.gt_json, preds, [1, 10, 100])
        ap300 = score(args.gt_json, preds, [1, 10, 300])
        print(f"{model:14s} AP@100={ap100:.4f}  AP@300={ap300:.4f}  "
              f"delta(+300)={ap300 - ap100:+.4f}  n_images={res['n_images']}",
              flush=True)


if __name__ == "__main__":
    main()
