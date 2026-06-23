"""Measure the Ultralytics validator's throughput gain from batching.

The portable yolo-validator is single-stream by design (one frame at a time, for
low peak memory + portability). The *Ultralytics* PyTorch validator, by contrast,
batches natively on the GPU — so this script sweeps the Ultralytics ``ult-pt`` lane
across batch sizes to quantify what that batching buys, against the batch=1 parity
baseline used everywhere else in this repo.

Letterbox is held at ``rect=False`` (square 640) so batch size is the only changed
variable: accuracy should stay ~identical across batches, and the FPS delta is the
pure batching effect (no rectangular-inference confound).

Usage::

    python -m benchmarks.batch_sweep \
        --models yolov8n yolov8n-seg --batches 1 8 16 32 --device 0 \
        --images-dir /home/sebastien/coco/val2017 \
        --gt-json /home/sebastien/coco/annotations/instances_val2017.json \
        --output-dir benchmarks/results/ult_batch_sweep

Run with ``YOLO_AUTOINSTALL=False`` to keep Ultralytics from re-installing
onnxruntime (see project setup notes).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from benchmarks.benchmark_a import _host_meta
from benchmarks.canonical_eval import canonical_eval
from benchmarks.coco_dataset import make_subset, setup_shadow
from benchmarks.runners import run_ultralytics


def _task_of(model_name: str) -> str:
    return "segment" if "seg" in model_name else "detect"


def sweep_model(model_name, run_yaml, run_gt_json, device, batches, rect, max_images):
    """Run the ult-pt lane across batch sizes for one model. Returns run dicts."""
    task = _task_of(model_name)
    iou_types = ("bbox", "segm") if task == "segment" else ("bbox",)
    pt = model_name if model_name.endswith(".pt") else f"{model_name}.pt"

    runs = []
    for batch in batches:
        print(f"\n[{model_name}] ult-pt batch={batch} (rect={rect}, device={device}) ...")
        try:
            res = run_ultralytics(pt, run_yaml, task, device=device,
                                  batch=batch, rect=rect)
            metrics = canonical_eval(run_gt_json, res["predictions"], iou_types)
            n, wall = res["n_images"], res["wall_s"]
            fps_wall = (n / wall) if (wall and n) else None
            run = {
                "batch": batch,
                "box_ap": metrics.get("bbox", {}).get("AP"),
                "mask_ap": metrics.get("segm", {}).get("AP") if task == "segment" else None,
                "speed_ms": res["speed"],          # {preprocess, inference, postprocess} per image
                "n_images": n,
                "wall_s": wall,
                "fps_wall": fps_wall,
            }
            box = run["box_ap"]
            print(f"  box AP={box:.4f}  inf={res['speed'].get('inference', 0):.2f} ms/img  "
                  f"wall={wall:.1f}s  fps_wall={fps_wall:.1f}")
            runs.append(run)
        except Exception as e:
            print(f"  [FAILED] batch={batch}: {e}")
            runs.append({"batch": batch, "error": str(e)})
    return task, runs


def _print_table(model_name, task, runs):
    base = next((r for r in runs if r.get("batch") == 1 and "error" not in r), None)
    base_fps = base["fps_wall"] if base else None
    seg = task == "segment"
    hdr = ["batch", "box AP"] + (["mask AP"] if seg else []) + \
          ["inf ms/img", "pre ms", "post ms", "wall FPS", "speedup"]
    print(f"\n## {model_name} ({task}) — ult-pt, rect=False")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in runs:
        if "error" in r:
            cells = [str(r["batch"]), f"ERROR: {r['error'][:40]}"]
            print("| " + " | ".join(cells) + " |")
            continue
        sp = r["speed_ms"]
        spd = (r["fps_wall"] / base_fps) if (base_fps and r["fps_wall"]) else None
        cells = [str(r["batch"]), f"{r['box_ap']:.4f}"]
        if seg:
            cells.append(f"{r['mask_ap']:.4f}" if r["mask_ap"] is not None else "—")
        cells += [
            f"{sp.get('inference', 0):.2f}",
            f"{sp.get('preprocess', 0):.2f}",
            f"{sp.get('postprocess', 0):.2f}",
            f"{r['fps_wall']:.1f}",
            f"{spd:.2f}×" if spd else "—",
        ]
        print("| " + " | ".join(cells) + " |")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["yolov8n", "yolov8n-seg"])
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 16, 32])
    p.add_argument("--device", default="0", help="CUDA index (e.g. 0) or 'cpu'")
    p.add_argument("--rect", action="store_true",
                   help="rectangular batches (default: square 640, isolates batch effect)")
    p.add_argument("--images-dir", default="/home/sebastien/coco/val2017")
    p.add_argument("--gt-json",
                   default="/home/sebastien/coco/annotations/instances_val2017.json")
    p.add_argument("--shadow-root", default="benchmarks/_coco_shadow")
    p.add_argument("--output-dir", default="benchmarks/results/ult_batch_sweep")
    p.add_argument("--max-images", type=int, default=None,
                   help="limit to first N images (smoke test); None = all 5000")
    a = p.parse_args()

    device = int(a.device) if a.device.isdigit() else a.device
    images_dir = Path(a.images_dir).expanduser()
    gt_json = Path(a.gt_json).expanduser()
    output_dir = Path(a.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shadow_root = setup_shadow(a.shadow_root, images_dir, gt_json)
    if a.max_images is not None:
        sub_images, sub_gt, sub_yaml = make_subset(shadow_root, images_dir, gt_json, a.max_images)
        run_yaml, run_gt = str(sub_yaml), str(sub_gt)
    else:
        run_yaml = str(shadow_root / "coco-val.yaml")
        run_gt = str(gt_json)

    print(f"Batch sweep — models {a.models}, batches {a.batches}, "
          f"device {device}, rect={a.rect}, max_images={a.max_images}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    for model_name in a.models:
        task, runs = sweep_model(model_name, run_yaml, run_gt, device,
                                 a.batches, a.rect, a.max_images)
        all_results.append({"model": model_name, "task": task, "runs": runs})

    out = {
        "lane": "ult-pt",
        "device": str(device),
        "rect": a.rect,
        "batches": a.batches,
        "max_images": a.max_images,
        "timestamp": timestamp,
        "host": _host_meta(),
        "models": all_results,
    }
    out_path = output_dir / f"batch_sweep_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")

    print("\n" + "=" * 60)
    print("ULTRALYTICS VALIDATOR — BATCH THROUGHPUT (ult-pt, CUDA)")
    print("=" * 60)
    for m in all_results:
        _print_table(m["model"], m["task"], m["runs"])


if __name__ == "__main__":
    main()
