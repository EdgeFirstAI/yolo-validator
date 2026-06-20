"""On-device HailoRT inference + COCO scoring for a compiled .hef (rpi5-hailo8l).

Runs on the Raspberry Pi 5 AI Kit (HailoRT installed). For each COCO val2017
image it letterboxes to the model input, runs the INT8 .hef (NMS is baked in by
the Dataflow Compiler), maps the per-class detections back to original-image
coordinates, and scores them with the same crowd-as-normal pycocotools path the
rest of the benchmark uses. Detection only.

This is yolo-validator filling the on-target gap: the Ultralytics/Vendor Hailo
workflow produces the .hef, but no consistent COCO mAP — this does. Output is a
``benchmark_a_<model>_<ts>.json`` (config key ``yv-hailo``) that
``benchmarks.normalize`` folds into ``metrics/rpi5-hailo8l.json`` unchanged.

Usage (on the Pi, repo importable, hailo_platform available)::

    python -m benchmarks.hailo_infer --hef yolo11n.hef --model yolo11n \
        --coco-val ~/coco/val2017 --gt ~/coco/annotations/instances_val2017.json \
        --out benchmarks/results/hailo
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

from benchmarks.canonical_eval import canonical_eval
from yolo_validator.coco_output import coco80_to_coco91


def _letterbox(img: Image.Image, size: int):
    """PIL RGB image -> (array[size,size,3] float32 0-255, scale, pad_x, pad_y).

    Same transform as the calibration/compile path so on-device input matches.
    """
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return np.asarray(canvas, dtype=np.float32), scale, pad_x, pad_y


def _decode_nms(raw, imgsz, scale, pad_x, pad_y, w0, h0, class_map, image_id,
                score_th):
    """Map a Hailo baked-NMS output to COCO prediction dicts (original coords).

    The yolov8 nms_postprocess output is a per-class list; each detection is
    ``[y_min, x_min, y_max, x_max, score]``. Coordinates are normalized [0,1]
    w.r.t. the letterboxed input (auto-detected: if values exceed ~1.5 they are
    already in input pixels). Un-letterbox -> original-image xywh.
    """
    recs = []
    # raw is a list/array indexed by class; each entry is (n_i, 5)
    for cls_idx, dets in enumerate(raw):
        if dets is None or len(dets) == 0:
            continue
        dets = np.asarray(dets, dtype=np.float32)
        for d in dets:
            y1, x1, y2, x2, score = float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4])
            if score < score_th:
                continue
            # normalized [0,1] -> input pixels (auto-detect already-pixel)
            mx = max(x1, x2, y1, y2)
            s = imgsz if mx <= 1.5 else 1.0
            x1, x2, y1, y2 = x1 * s, x2 * s, y1 * s, y2 * s
            # un-letterbox -> original image coords
            ox1 = (x1 - pad_x) / scale
            oy1 = (y1 - pad_y) / scale
            ox2 = (x2 - pad_x) / scale
            oy2 = (y2 - pad_y) / scale
            ox1, ox2 = max(0.0, min(ox1, w0)), max(0.0, min(ox2, w0))
            oy1, oy2 = max(0.0, min(oy1, h0)), max(0.0, min(oy2, h0))
            bw, bh = ox2 - ox1, oy2 - oy1
            if bw <= 0 or bh <= 0:
                continue
            recs.append({
                "image_id": int(image_id),
                "category_id": int(class_map[cls_idx]),
                "bbox": [ox1, oy1, bw, bh],
                "score": score,
            })
    return recs


def run(hef_path, model, coco_val, gt, out_dir, imgsz, limit, score_th):
    from hailo_platform import (HEF, ConfigureParams, FormatType,
                                HailoStreamInterface, InferVStreams,
                                InputVStreamParams, OutputVStreamParams, VDevice)

    hef = HEF(str(hef_path))
    in_info = hef.get_input_vstream_infos()[0]
    in_name = in_info.name
    class_map = coco80_to_coco91()

    images = sorted(Path(coco_val).glob("*.jpg"))
    if limit:
        images = images[:limit]
    print(f"[hailo_infer] {model}: {len(images)} images, hef={Path(hef_path).name}")

    preds, t_pre, t_inf, t_post = [], 0.0, 0.0, 0.0
    params = VDevice.create_params()
    with VDevice(params) as target:
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng = target.configure(hef, cfg)[0]
        ng_params = ng.create_params()
        in_params = InputVStreamParams.make(ng, quantized=False, format_type=FormatType.FLOAT32)
        out_params = OutputVStreamParams.make(ng, quantized=False, format_type=FormatType.FLOAT32)
        wall0 = time.perf_counter()
        with InferVStreams(ng, in_params, out_params) as pipeline:
            # Vendor Model-Zoo HEFs bake a *deployment* NMS score threshold
            # (e.g. 0.2) that clips low-confidence boxes and depresses COCO mAP.
            # Override to the eval threshold at runtime so the score is
            # comparable to the other lanes (no-op / ignored for HEFs without a
            # HAILO_NMS output, e.g. NMS-free yolo26).
            try:
                pipeline.set_nms_score_threshold(score_th)
            except Exception:
                pass
            with ng.activate(ng_params):
                for i, img_path in enumerate(images):
                    image_id = int(img_path.stem)
                    t0 = time.perf_counter()
                    with Image.open(img_path) as im:
                        im = im.convert("RGB")
                        w0, h0 = im.size
                        arr, scale, pad_x, pad_y = _letterbox(im, imgsz)
                    inp = {in_name: arr[np.newaxis, ...]}
                    t1 = time.perf_counter()
                    results = pipeline.infer(inp)
                    t2 = time.perf_counter()
                    raw = results[next(iter(results))][0]  # batch 0 -> per-class list
                    preds.extend(_decode_nms(raw, imgsz, scale, pad_x, pad_y,
                                             w0, h0, class_map, image_id, score_th))
                    t3 = time.perf_counter()
                    t_pre += t1 - t0
                    t_inf += t2 - t1
                    t_post += t3 - t2
                    if (i + 1) % 500 == 0:
                        print(f"  {i + 1}/{len(images)} ({len(preds)} dets)")
        wall = time.perf_counter() - wall0

    n = len(images)
    print(f"[hailo_infer] scoring {len(preds)} detections over {n} images…")
    metrics = canonical_eval(str(gt), preds, iou_types=("bbox",))

    cfg = {
        "bbox": metrics["bbox"],
        "segm": None,
        "timing": {
            "preprocess": 1e3 * t_pre / n,
            "inference": 1e3 * t_inf / n,
            "postprocess": 1e3 * t_post / n,
            "e2e": 1e3 * (t_pre + t_inf + t_post) / n,
        },
        "fps_wall": n / wall if wall else None,
        "n_images": n,
    }
    doc = {
        "label": model, "task": "detect",
        "host": {"machine": platform.machine(), "node": platform.node(),
                 "system": platform.platform(), "device": "rpi5-hailo8l"},
        "configs": {"yv-hailo": cfg},
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"benchmark_a_{model}_{ts}.json"
    out_path.write_text(json.dumps(doc, indent=2))
    b = metrics["bbox"]
    print(f"[hailo_infer] {model}: box AP={b['AP']:.4f} AP50={b['AP50']:.4f} "
          f"| {cfg['fps_wall']:.1f} fps | inf {cfg['timing']['inference']:.2f} ms "
          f"-> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hef", required=True, type=Path)
    ap.add_argument("--model", required=True, help="variant label, e.g. yolo11n")
    ap.add_argument("--coco-val", type=Path, default=Path.home() / "coco" / "val2017")
    ap.add_argument("--gt", type=Path,
                    default=Path.home() / "coco" / "annotations" / "instances_val2017.json")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results/hailo"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0, help="0 = all val2017")
    ap.add_argument("--score-th", type=float, default=0.001)
    a = ap.parse_args()
    run(a.hef, a.model, a.coco_val, a.gt, a.out, a.imgsz, a.limit, a.score_th)


if __name__ == "__main__":
    main()
