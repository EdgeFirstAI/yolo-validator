"""CLI mirroring Ultralytics `val` semantics so one config drives both."""
from __future__ import annotations

import argparse
import glob
import json
import os
import zlib

from ._stats import format_stage_table
from .backends import load_backend
from .coco_eval import evaluate_coco
from .coco_output import coco80_to_coco91, detections_to_coco
from .pipeline import ValidationPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="yolo-validator", description="Standalone YOLO reference validator")
    p.add_argument("--model", required=True)
    p.add_argument("--images", required=True, help="image dir or glob")
    p.add_argument("--gt", default=None, help="COCO GT json for evaluation")
    p.add_argument("--runtime", default="onnx")
    p.add_argument("--task", default="auto", choices=["auto", "detect", "segment"])
    p.add_argument("--preprocess", default="auto", choices=["auto", "torch", "numpy"])
    p.add_argument("--postprocess", default="auto", choices=["auto", "torch", "numpy"])
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--batch", type=int, default=1,
                   help="Inference batch size. 1 = single-stream latency reference; "
                        "N>1 measures batched throughput and needs a dynamic-batch ONNX.")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--output", default="results.json")
    p.add_argument("--provider", default="cpu", help="onnx EP: cpu|cuda|coreml")
    p.add_argument("--e2e", dest="e2e", action=argparse.BooleanOptionalAction, default=None,
                   help="Force NMS-free/end-to-end decode (--e2e) or classic anchor-grid "
                        "(--no-e2e). Default: auto-detect from output shape.")
    return p


def _list_images(spec: str) -> list[str]:
    if os.path.isdir(spec):
        files: list[str] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files += glob.glob(os.path.join(spec, ext))
        return sorted(files)
    return sorted(glob.glob(spec))


def _image_id(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem)
    except ValueError:
        # Deterministic across runs (Python's hash() is salted by
        # PYTHONHASHSEED). Non-numeric stems still won't match COCO GT ids,
        # but at least a run is reproducible against itself.
        return zlib.crc32(stem.encode("utf-8"))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # Validate the image set first (fail fast, before loading a model).
    images = _list_images(args.images)
    if args.max_images is not None:
        images = images[: args.max_images]
    if not images:
        raise SystemExit(f"no images found at {args.images!r}")
    if len(images) <= args.warmup:
        raise SystemExit(
            f"need more than --warmup ({args.warmup}) images to measure; "
            f"got {len(images)}. Lower --warmup or provide more images."
        )

    task = None if args.task == "auto" else args.task
    backend = load_backend(args.model, runtime=args.runtime, provider=args.provider,
                           task=task, e2e=args.e2e)
    pipe = ValidationPipeline(
        backend, preprocess_path=args.preprocess, postprocess_path=args.postprocess,
        conf=args.conf, iou=args.iou, max_det=args.max_det,
        with_masks=(backend.spec.task == "segment"),
    )

    class_map = coco80_to_coco91()
    predictions: list[dict] = []

    def on_frame(res):
        predictions.extend(
            detections_to_coco(_image_id(res.image_path), res.detections, class_map,
                               masks=res.masks if len(res.masks) else None)
        )

    stats = pipe.run(images, warmup=args.warmup, on_frame=on_frame, batch_size=args.batch)

    print(format_stage_table(stats.stages))
    e2e = sum(s.mean_ms for k, s in stats.stages.items()
              if k in ("preprocess", "inference", "decode", "mask"))
    fps = f"{1000.0 / e2e:.1f} FPS" if e2e > 0 else "N/A"
    print(f"\nModel-path (pre+inf+post) mean: {e2e:.2f} ms  ({fps})  "
          f"over {stats.n_images} images")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    if args.gt:
        iou_types = ("bbox", "segm") if backend.spec.task == "segment" else ("bbox",)
        res = evaluate_coco(args.gt, predictions, iou_types=iou_types)
        for t in iou_types:
            print(f"{t}: AP={res[t]['AP']:.4f}  AP50={res[t]['AP50']:.4f}  AR100={res[t]['AR100']:.4f}")
    return 0
