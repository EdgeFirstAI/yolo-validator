"""On-device TFLite INT8 inference + COCO scoring for the NXP NPU targets.

The Ultralytics/Vendor workflow produces a full-integer INT8 ``.tflite`` (see
``export_tflite_int8.py``); this runs it on-target and produces a consistent
COCO mAP the way the rest of the benchmark does — so platforms without bespoke
vendor tooling (unlike Hailo's tuned HEFs) get the same crowd-as-normal score.

Runs the model via the TFLite ``Interpreter`` with an optional hardware delegate:
  - i.MX 8M Plus: VX delegate (``/usr/lib/libvx_delegate.so``)
  - i.MX 95:      Neutron delegate (``/usr/lib/libneutron_delegate.so``), on the
                  Neutron-converted model
  - host smoke:   ``--delegate none`` (CPU reference, via TensorFlow's tf.lite)

The Ultralytics INT8 TFLite output is already decoded (box + sigmoid'd class
scores, NMS-free) — so decode here is just dequantize → split → class-aware NMS →
(seg) mask materialize. Emits a ``benchmark_a_<model>_<ts>.json`` (config key
``yv-tflite``) that ``benchmarks.normalize`` folds into ``metrics/<platform>.json``.

Usage (on-device)::

    python3 -m benchmarks.tflite_infer --tflite yolov8n-int8.tflite --model yolov8n \
        --delegate /usr/lib/libvx_delegate.so --device-label imx8mp-vsi \
        --coco-val ~/coco/val2017 --gt ~/coco/annotations/instances_val2017.json
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import time
from pathlib import Path

import cv2
import numpy as np

from benchmarks.canonical_eval import canonical_eval
from yolo_validator.coco_output import coco80_to_coco91, detections_to_coco
from yolo_validator.detections import Detections
from yolo_validator.letterbox import LetterboxInfo, unletterbox_boxes
from yolo_validator.masks import materialize_masks_numpy
from yolo_validator.nms import nms_class_aware

_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _safe_label(name: str) -> str:
    if not _LABEL_RE.fullmatch(name) or ".." in name:
        raise SystemExit(f"invalid model label {name!r}: expected [A-Za-z0-9._-]")
    return name


def _load_interpreter(model_path: str, delegate: str | None):
    """Build a TFLite Interpreter (tflite_runtime on-device, tf.lite on host)."""
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError:  # host smoke (venv-tfexport): ai_edge_litert ships the runtime
        from ai_edge_litert.interpreter import Interpreter, load_delegate
    delegates = []
    if delegate and delegate != "none":
        print(f"[tflite] loading delegate {delegate}")
        delegates = [load_delegate(delegate)]
    interp = Interpreter(model_path=model_path, experimental_delegates=delegates)
    interp.allocate_tensors()
    return interp


def _letterbox(path: Path, size: int):
    """JPEG decode + letterbox → (uint8 RGB [size,size,3], LetterboxInfo)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise RuntimeError(f"failed to decode {path}")
    h0, w0 = img.shape[:2]
    scale = min(size / w0, size / h0)
    nw, nh = round(w0 * scale), round(h0 * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas, LetterboxInfo(scale, pad_x, pad_y, w0, h0)


def _quantize_input(img_rgb_u8: np.ndarray, inp_detail) -> np.ndarray:
    """Normalize to [0,1] then quantize to the model's input dtype (or keep float)."""
    x = img_rgb_u8.astype(np.float32) / 255.0
    dt = inp_detail["dtype"]
    if np.issubdtype(dt, np.integer):
        scale, zp = inp_detail["quantization"]
        if scale == 0:
            scale = 1.0
        q = np.round(x / scale + zp)
        info = np.iinfo(dt)
        q = np.clip(q, info.min, info.max).astype(dt)
        return q[None, ...]
    return x.astype(np.float32)[None, ...]


def _dequant(arr: np.ndarray, detail) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.integer):
        scale, zp = detail["quantization"]
        return (arr.astype(np.float32) - zp) * scale
    return arr.astype(np.float32)


def _orient_head(a: np.ndarray, ch: int) -> np.ndarray:
    """Return (N, ch) from a (1, ch, N) or (1, N, ch) detection-head tensor."""
    a = a[0]
    return a.T if a.shape[0] == ch else a


def run(tflite_path, model, coco_val, gt, out_dir, device_label, delegate,
        imgsz, limit, score_th, iou=0.7, max_det=300, top_k=1000):
    model = _safe_label(model)
    tflite_path = Path(tflite_path).expanduser().resolve()
    coco_val = Path(coco_val).expanduser().resolve()
    gt = Path(gt).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    for p in (tflite_path, gt):
        if not p.exists():
            raise SystemExit(f"not found: {p}")

    interp = _load_interpreter(str(tflite_path), delegate)
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()
    class_map = coco80_to_coco91()

    images = sorted(coco_val.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no *.jpg under {coco_val}")
    if limit:
        images = images[:limit]
    seg = "seg" in model
    print(f"[tflite_infer] {model}: {len(images)} images, {tflite_path.name}, "
          f"task={'segment' if seg else 'detect'}, in={inp['dtype'].__name__} "
          f"{inp['shape']}, {len(outs)} outputs")

    preds, t_pre, t_inf, t_post = [], 0.0, 0.0, 0.0
    wall0 = time.perf_counter()
    for i, img_path in enumerate(images):
        image_id = int(img_path.stem)
        t0 = time.perf_counter()
        canvas, lb = _letterbox(img_path, imgsz)
        x = _quantize_input(canvas, inp)
        t1 = time.perf_counter()
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        raw = [(_dequant(interp.get_tensor(o["index"]), o)) for o in outs]
        t2 = time.perf_counter()
        # end2end (NMS-free) head emits a single [1, N, 6] = [x1,y1,x2,y2,score,class]
        # tensor (boxes normalized in letterbox space, already top-k/NMS-free): decode
        # directly, no class-aware NMS. Classic head keeps the [1, ch, anchors] grid.
        e2e_t = None if seg else next(
            (o for o in raw if o.ndim == 3 and o.shape[-1] == 6), None)
        if e2e_t is not None:
            arr = e2e_t[0]                               # (N, 6)
            xyxy = arr[:, :4].astype(np.float32)
            sc = arr[:, 4].astype(np.float32)
            cls = np.rint(arr[:, 5]).astype(np.int64)
            if xyxy.size and float(xyxy.max()) <= 1.5:   # normalized -> pixel
                xyxy = xyxy * imgsz
            keep = np.nonzero(sc > score_th)[0]
            keep = keep[np.argsort(-sc[keep])][:max_det]
            if keep.size:
                det = Detections(boxes=unletterbox_boxes(xyxy[keep], lb),
                                 scores=sc[keep], classes=cls[keep])
                preds.extend(detections_to_coco(image_id, det, class_map, None))
        else:
            ch = 116 if seg else 84
            det_t = next(o for o in raw if ch in o.shape)
            head = _orient_head(det_t, ch)               # (N, ch)
            boxes_xywh, scores = head[:, :4], head[:, 4:84]
            coeffs = head[:, 84:116] if seg else None
            # auto-detect normalized vs pixel box coords
            if boxes_xywh.size and float(boxes_xywh.max()) <= 1.5:
                boxes_xywh = boxes_xywh * imgsz
            cx, cy, w, h = boxes_xywh.T
            xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
            flat = scores.reshape(-1)
            keep = np.nonzero(flat > score_th)[0]
            if keep.size:
                if keep.size > top_k:
                    keep = keep[np.argpartition(flat[keep], -top_k)[-top_k:]]
                nc = scores.shape[1]
                anc, cl, sc = keep // nc, keep % nc, flat[keep]
                box_lb = xyxy[anc]
                idx = nms_class_aware(box_lb, sc, cl, iou)[:max_det]
                box_lb = box_lb[idx]
                cls = cl[idx].astype(np.int64)
                sc = sc[idx]
                if seg:
                    proto = next(o[0] for o in raw if o.ndim == 4 and 32 in o.shape)
                    proto = proto.transpose(2, 0, 1) if proto.shape[-1] == 32 else proto
                    det = Detections(boxes=unletterbox_boxes(box_lb, lb), scores=sc,
                                     classes=cls, coeffs=coeffs[anc][idx],
                                     protos=proto[None].astype(np.float32), boxes_lb=box_lb)
                    masks = materialize_masks_numpy(det.protos, det.coeffs, det.boxes_lb,
                                                    lb, imgsz, imgsz) if len(sc) else []
                    preds.extend(detections_to_coco(image_id, det, class_map, masks))
                else:
                    det = Detections(boxes=unletterbox_boxes(box_lb, lb), scores=sc,
                                     classes=cls)
                    preds.extend(detections_to_coco(image_id, det, class_map, None))
        t3 = time.perf_counter()
        t_pre += t1 - t0; t_inf += t2 - t1; t_post += t3 - t2
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(images)} ({len(preds)} dets)")
    wall = time.perf_counter() - wall0

    n = len(images)
    iou_types = ("bbox", "segm") if seg else ("bbox",)
    print(f"[tflite_infer] scoring {len(preds)} dets over {n} images…")
    metrics = canonical_eval(str(gt), preds, iou_types=iou_types)
    cfg = {
        "bbox": metrics["bbox"], "segm": metrics.get("segm"),
        "timing": {"preprocess": 1e3 * t_pre / n, "inference": 1e3 * t_inf / n,
                   "postprocess": 1e3 * t_post / n,
                   "e2e": 1e3 * (t_pre + t_inf + t_post) / n},
        "fps_wall": n / wall if wall else None, "n_images": n,
    }
    doc = {"label": model, "task": "segment" if seg else "detect",
           "host": {"machine": platform.machine(), "node": platform.node(),
                    "system": platform.platform(), "device": device_label},
           "configs": {"yv-tflite": cfg}}
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"benchmark_a_{model}_{ts}.json"
    out_path.write_text(json.dumps(doc, indent=2))
    b = metrics["bbox"]
    mask = f" mask AP={metrics['segm']['AP']:.4f}" if seg and metrics.get("segm") else ""
    print(f"[tflite_infer] {model}: box AP={b['AP']:.4f} AP50={b['AP50']:.4f}{mask} "
          f"| {cfg['fps_wall']:.2f} fps | inf {cfg['timing']['inference']:.1f} ms -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tflite", required=True, type=Path)
    ap.add_argument("--model", required=True, help="variant label, e.g. yolov8n")
    ap.add_argument("--delegate", default="none",
                    help="path to delegate .so, or 'none' for CPU")
    ap.add_argument("--device-label", required=True,
                    help="metrics platform id, e.g. imx8mp-vsi / imx95-neutron")
    ap.add_argument("--coco-val", type=Path, default=Path.home() / "coco" / "val2017")
    ap.add_argument("--gt", type=Path,
                    default=Path.home() / "coco" / "annotations" / "instances_val2017.json")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results/tflite"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0, help="0 = all val2017")
    ap.add_argument("--score-th", type=float, default=0.001)
    a = ap.parse_args()
    run(a.tflite, a.model, a.coco_val, a.gt, a.out, a.device_label, a.delegate,
        a.imgsz, a.limit, a.score_th)


if __name__ == "__main__":
    main()
