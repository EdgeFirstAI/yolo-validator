# benchmarks/parity_measure.py
"""Isolate the Ultralytics-vs-EdgeFirst accuracy methodology delta on COCO val2017.

For each model this runs the Ultralytics validator (YOLO.val on the official .pt)
TWICE — rect=False (square 640 letterbox, what the repo's baseline uses) and
rect=True (rectangular inference, the Ultralytics default) — then scores each
prediction set through the repo's own ``canonical_eval`` in BOTH crowd modes:

  * crowd-as-normal (ignore_crowd=False) — force iscrowd=0, how EdgeFirst scores.
  * crowd-ignored   (ignore_crowd=True)  — standard COCO / Ultralytics.

So per model we get a 2x2 grid of box mAP@0.5:0.95 (and mask mAP for -seg):

              crowd-as-normal      crowd-ignored
  rect=False        A                   B
  rect=True         C                   D   <- the Ultralytics-parity number

  A  ~= the repo's existing ult baseline (onnx-cuda.json, validator=ultralytics)
  D   = Ultralytics-parity (rect=True + crowd-ignored)
  C-A = rect effect (rect=True vs rect=False), crowd held at as-normal
  B-A = crowd effect (ignore vs force-0),      rect held at False

This touches NO EdgeFirst Studio; it is compute-only and reuses the exact
``run_ultralytics`` + ``canonical_eval`` functions the harness uses.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmarks.canonical_eval import canonical_eval
from benchmarks.runners import run_ultralytics

REPO = Path(__file__).resolve().parent.parent

# model -> (task, classic-head?). classic head means Detect.end2end=False.
DEFAULT_MODELS = [
    "yolov8n", "yolo11n", "yolo26n", "yolov5nu",
    "yolov8n-seg", "yolo11n-seg", "yolo26n-seg",
]


def _load_model(pt_path: str):
    """Load a YOLO .pt and force the CLASSIC (one2many) head (end2end=False)."""
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect
    m = YOLO(pt_path)
    for mod in m.model.modules():
        if isinstance(mod, Detect):
            mod.end2end = False  # classic head (yolo26 defaults to nms-free)
            break
    return m


def _ap(metrics: dict, iou_type: str) -> float:
    return float((metrics.get(iou_type) or {}).get("AP", float("nan")))


def measure_model(model: str, data_yaml: str, gt_json: str, device) -> dict:
    task = "segment" if model.endswith("-seg") else "detect"
    iou_types = ("bbox", "segm") if task == "segment" else ("bbox",)
    pt_path = str(REPO / f"{model}.pt")

    out: dict = {"model": model, "task": task, "grid": {}}
    for rect in (False, True):
        t0 = time.perf_counter()
        pre = _load_model(pt_path)
        res = run_ultralytics(pt_path, data_yaml, task, pre_val_model=pre,
                              device=device, rect=rect)
        wall = time.perf_counter() - t0
        preds = res["predictions"]
        out["n_images"] = res["n_images"]
        for ignore_crowd in (False, True):
            m = canonical_eval(gt_json, preds, iou_types, ignore_crowd=ignore_crowd)
            key = f"rect={rect},crowd={'ignored' if ignore_crowd else 'as-normal'}"
            entry = {"box_ap": _ap(m, "bbox")}
            if task == "segment":
                entry["mask_ap"] = _ap(m, "segm")
            out["grid"][key] = entry
        print(f"  [{model}] rect={rect} wall={wall:.0f}s "
              f"box(as-normal)={out['grid'][f'rect={rect},crowd=as-normal']['box_ap']:.4f} "
              f"box(ignored)={out['grid'][f'rect={rect},crowd=ignored']['box_ap']:.4f}",
              flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-yaml",
                   default=str(REPO / "benchmarks/_coco_shadow/coco-val.yaml"))
    p.add_argument("--gt-json",
                   default=str(Path("~/coco/annotations/instances_val2017.json").expanduser()))
    p.add_argument("--out", default=str(REPO / "benchmarks/results/parity_measure.json"))
    args = p.parse_args()

    device = 0 if args.device == "cuda" else args.device
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"parity_measure: models={args.models} device={args.device}")
    print(f"  data_yaml={args.data_yaml}")
    print(f"  gt_json  ={args.gt_json}\n")

    results = []
    for model in args.models:
        print(f"=== {model} ===", flush=True)
        try:
            r = measure_model(model, args.data_yaml, args.gt_json, device)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = {"model": model, "error": str(e)}
        results.append(r)
        Path(args.out).write_text(json.dumps(results, indent=1))  # checkpoint each model

    print("\n==== SUMMARY (box mAP@0.5:0.95; mask in parens for -seg) ====")
    for r in results:
        if "error" in r:
            print(f"{r['model']:14s} ERROR {r['error']}")
            continue
        g = r["grid"]
        def cell(k):
            e = g[k]
            return f"{e['box_ap']:.4f}" + (f"/{e['mask_ap']:.4f}" if 'mask_ap' in e else "")
        print(f"{r['model']:14s} "
              f"A[rF,cN]={cell('rect=False,crowd=as-normal')} "
              f"B[rF,cI]={cell('rect=False,crowd=ignored')} "
              f"C[rT,cN]={cell('rect=True,crowd=as-normal')} "
              f"D[rT,cI]={cell('rect=True,crowd=ignored')}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
