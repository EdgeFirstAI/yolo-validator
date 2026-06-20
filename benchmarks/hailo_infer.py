"""On-device HailoRT inference + COCO scoring for a compiled .hef (rpi5-hailo8l).

Runs on the Raspberry Pi 5 AI Kit (HailoRT installed). For each COCO val2017
image it letterboxes to the model input, runs the INT8 .hef, maps detections
back to original-image coordinates, and scores them with the same
crowd-as-normal pycocotools path the rest of the benchmark uses.

Three head types are auto-dispatched by output-vstream count (override with
``--head``): baked HAILO_NMS detect (decode is on-chip), raw NMS-free detect
(yolo26 — anchor-free distance box + class logits decoded here), and raw
yolov8 instance segmentation (DFL box + class + mask coeffs + prototype,
decoded and masked here). See ``_decode_nms`` / ``_decode_nmsfree`` /
``_decode_seg``.

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
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

from benchmarks.canonical_eval import canonical_eval
from yolo_validator.coco_output import coco80_to_coco91, detections_to_coco
from yolo_validator.letterbox import LetterboxInfo

_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _safe_label(name: str) -> str:
    """Reject model labels with path/shell metacharacters before they reach a
    filename (defense-in-depth for untrusted CLI input — no separators, no
    leading dash, no ``..`` traversal)."""
    if not _LABEL_RE.fullmatch(name) or ".." in name:
        raise SystemExit(f"invalid model label {name!r}: expected [A-Za-z0-9._-]")
    return name


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


def _sigmoid(x):
    # clip avoids exp overflow on large-negative class logits (-> 0.0 anyway)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _decode_nmsfree(results, imgsz, scale, pad_x, pad_y, w0, h0, class_map,
                    image_id, score_th, max_det=300):
    """Decode a raw NMS-free head (yolo26) to COCO dicts.

    The HEF emits, per stride, a 4-channel box branch (anchor-free *distances*
    l,t,r,b in grid units) and an nc-channel class-logit branch — no DFL, no
    on-chip NMS. Decode = cell-center (+0.5) anchor · stride, sigmoid on class,
    multi-label top-k, and (because the one-to-one head suppresses duplicates)
    NO NMS — matching how Ultralytics validates yolov10/yolo26.
    """
    grids = {}
    for v in results.values():
        a = np.asarray(v)[0]  # (H, W, C)
        h, _, c = a.shape
        grids.setdefault(h, {})[("box" if c == 4 else "cls")] = a
    boxes_all, scores_all = [], []
    for h, g in grids.items():
        stride = imgsz // h
        box = g["box"].reshape(-1, 4)
        cls = _sigmoid(g["cls"].reshape(-1, g["cls"].shape[-1]))
        gx, gy = np.meshgrid(np.arange(h) + 0.5, np.arange(h) + 0.5)
        ax, ay = gx.reshape(-1), gy.reshape(-1)
        l, t, r, b = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        boxes_all.append(np.stack([(ax - l) * stride, (ay - t) * stride,
                                   (ax + r) * stride, (ay + b) * stride], 1))
        scores_all.append(cls)
    boxes = np.concatenate(boxes_all, 0)        # (8400, 4) xyxy, input space
    scores = np.concatenate(scores_all, 0)      # (8400, nc)
    nc = scores.shape[1]
    flat = scores.reshape(-1)
    keep = np.nonzero(flat > score_th)[0]
    if keep.size == 0:
        return []
    if keep.size > max_det:
        keep = keep[np.argpartition(flat[keep], -max_det)[-max_det:]]
    anc, cls_idx, sc = keep // nc, keep % nc, flat[keep]
    recs = []
    for k in range(keep.size):
        x1, y1, x2, y2 = boxes[anc[k]]
        ox1, oy1 = (x1 - pad_x) / scale, (y1 - pad_y) / scale
        ox2, oy2 = (x2 - pad_x) / scale, (y2 - pad_y) / scale
        ox1, ox2 = max(0.0, min(ox1, w0)), max(0.0, min(ox2, w0))
        oy1, oy2 = max(0.0, min(oy1, h0)), max(0.0, min(oy2, h0))
        if ox2 - ox1 <= 0 or oy2 - oy1 <= 0:
            continue
        recs.append({"image_id": int(image_id),
                     "category_id": int(class_map[int(cls_idx[k])]),
                     "bbox": [ox1, oy1, ox2 - ox1, oy2 - oy1],
                     "score": float(sc[k])})
    return recs


def _dfl(box):
    """DFL: (N,64) box logits -> (N,4) integrated distances (softmax over 16)."""
    x = box.reshape(-1, 4, 16)
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(-1, keepdims=True)
    return (p * np.arange(16, dtype=np.float32)).sum(-1)


def _fast_masks(protos, coeffs, boxes_lb, lb, imgsz):
    """Fast instance-mask materialization for the Pi's ARM CPU.

    The repo's native path upsamples every 160² prototype mask to the full input
    resolution (640²) in a Python/cv2 loop — ~30 ms/mask on ARM, untenable at
    100 masks/image. This is the Ultralytics ``process_mask`` order instead:
    sigmoid(proto·coeffs) at proto res, crop to the box at proto res
    (vectorized), strip the letterbox pad, then a SINGLE small resize (160→orig)
    per mask. ~6-10× faster; mask mAP within ~0.5 pp of native.
    """
    import cv2

    if coeffs is None or len(coeffs) == 0:
        return []
    c, mh, mw = protos.shape[1], protos.shape[2], protos.shape[3]
    proto = protos[0].astype(np.float32).reshape(c, -1)
    m = _sigmoid((coeffs.astype(np.float32) @ proto).reshape(-1, mh, mw))  # (N,mh,mw)
    sx, sy = mw / imgsz, mh / imgsz                      # input -> proto scale
    b = np.asarray(boxes_lb, np.float32)
    cols, rows = np.arange(mw)[None, None, :], np.arange(mh)[None, :, None]
    x1, x2 = (b[:, 0] * sx)[:, None, None], (b[:, 2] * sx)[:, None, None]
    y1, y2 = (b[:, 1] * sy)[:, None, None], (b[:, 3] * sy)[:, None, None]
    m = m * ((cols >= x1) & (cols < x2) & (rows >= y1) & (rows < y2))
    left, top = int(round(lb.pad_x * sx)), int(round(lb.pad_y * sy))
    right, bottom = mw - left, mh - top
    out = []
    for mi in m:
        crop = mi[top:bottom, left:right]
        if crop.size == 0:
            out.append(np.zeros((lb.orig_h, lb.orig_w), np.uint8))
            continue
        rm = cv2.resize(crop, (lb.orig_w, lb.orig_h), interpolation=cv2.INTER_LINEAR)
        out.append((rm > 0.5).astype(np.uint8))
    return out


def _decode_seg(results, imgsz, lb, score_th, iou=0.7, max_det=100, top_k=1000):
    """Decode raw yolov8-seg outputs -> (Detections, masks).

    The HEF emits, per stride, a 64-ch DFL box branch + nc-ch class logits +
    nm-ch mask coefficients, plus an (mh, mw, nm) prototype tensor — no on-chip
    decode/NMS/masks. We rebuild the standard ``[1, 4+nc+nm, 8400]`` array
    (DFL+anchor box decode -> xywh, sigmoid class, raw coeffs) and a
    ``[1, nm, mh, mw]`` proto, then reuse yolo-validator's VALIDATED
    NumpyPostprocessor (conf + class-aware NMS) and materialize_masks_numpy
    (σ(proto·coeffs), crop, un-letterbox) so masks match the other lanes.
    """
    from yolo_validator.detections import Detections
    from yolo_validator.letterbox import unletterbox_boxes
    from yolo_validator.nms import nms_class_aware

    proto, per = None, {}
    for v in results.values():
        a = np.asarray(v)[0]            # (H, W, C)
        h, _, c = a.shape
        if c == 32 and h >= 160:
            proto = a                   # (mh, mw, nm) prototype masks
        else:
            per.setdefault(h, {})[c] = a
    if proto is None:
        raise ValueError(
            "segment head selected but no prototype tensor (expected a "
            "(>=160, >=160, 32) output) was found among the HEF outputs — "
            "this HEF is not a yolov8-seg model. Pass the correct --head.")
    boxes, clss, coeffs = [], [], []
    for h in sorted(per, reverse=True):
        g, stride = per[h], imgsz // h
        d = _dfl(g[64].reshape(-1, 64))     # (HW, 4) distances l,t,r,b
        gx, gy = np.meshgrid(np.arange(h) + 0.5, np.arange(h) + 0.5)
        ax, ay = gx.reshape(-1), gy.reshape(-1)
        l, t, r, b = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
        boxes.append(np.stack([(ax - l) * stride, (ay - t) * stride,
                               (ax + r) * stride, (ay + b) * stride], 1))   # xyxy
        clss.append(_sigmoid(g[80].reshape(-1, 80)))
        coeffs.append(g[32].reshape(-1, 32))
    boxes = np.concatenate(boxes, 0)            # (8400, 4) xyxy input space
    cls = np.concatenate(clss, 0)               # (8400, nc)
    coef = np.concatenate(coeffs, 0)            # (8400, nm)
    proto_t = proto.transpose(2, 0, 1)[None].astype(np.float32)   # (1, nm, mh, mw)
    nm = coef.shape[1]
    # Multi-label candidates (anchor×class), capped to top-K by score BEFORE the
    # O(n^2) greedy NMS. The ARM CPU chokes on the ~30k pairs a 0.001 threshold
    # yields (the detect lanes dodge this via the device's baked C NMS); top-K
    # >> COCO maxDets=100, so this is mAP-neutral.
    flat = cls.reshape(-1)
    keep = np.nonzero(flat > score_th)[0]
    if keep.size == 0:
        empty = Detections(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                           np.zeros((0,), np.int64), np.zeros((0, nm), np.float32),
                           proto_t, np.zeros((0, 4), np.float32))
        return empty, []
    if keep.size > top_k:
        keep = keep[np.argpartition(flat[keep], -top_k)[-top_k:]]
    anc, cl, sc = keep // cls.shape[1], keep % cls.shape[1], flat[keep]
    xyxy = boxes[anc]
    idx = nms_class_aware(xyxy, sc, cl, iou)[:max_det]
    boxes_lb = xyxy[idx]
    det = Detections(boxes=unletterbox_boxes(boxes_lb, lb), scores=sc[idx],
                     classes=cl[idx].astype(np.int64), coeffs=coef[anc][idx],
                     protos=proto_t, boxes_lb=boxes_lb)
    masks = _fast_masks(det.protos, det.coeffs, det.boxes_lb, lb, imgsz) if len(det.scores) else []
    return det, masks


def run(hef_path, model, coco_val, gt, out_dir, imgsz, limit, score_th,
        head="auto"):
    from hailo_platform import (HEF, ConfigureParams, FormatType,
                                HailoStreamInterface, InferVStreams,
                                InputVStreamParams, OutputVStreamParams, VDevice)

    # The usage string advertises ``~/...`` paths; Path() does not expand ``~``,
    # so expand + resolve (the latter normalizes away any ``..`` traversal) and
    # fail fast rather than silently scoring 0 images. ``model`` is validated
    # because it is interpolated into the output filename below.
    model = _safe_label(model)
    hef_path = Path(hef_path).expanduser().resolve()
    coco_val = Path(coco_val).expanduser().resolve()
    gt = Path(gt).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if not hef_path.exists():
        raise SystemExit(f"HEF not found: {hef_path}")
    if not gt.exists():
        raise SystemExit(f"COCO ground-truth not found: {gt}")

    hef = HEF(str(hef_path))
    in_info = hef.get_input_vstream_infos()[0]
    in_name = in_info.name
    class_map = coco80_to_coco91()

    # Head type from the HEF's output vstreams: 1 = baked HAILO_NMS (yolov8/11/
    # v5 detect); 6 = raw NMS-free (yolo26: 3×box + 3×cls); else raw seg.
    n_out = len(hef.get_output_vstream_infos())
    if head == "auto":
        head = "nms" if n_out == 1 else "nmsfree" if n_out == 6 else "segment"

    images = sorted(Path(coco_val).glob("*.jpg"))
    if not images:
        raise SystemExit(f"no *.jpg images found under {coco_val} — check --coco-val")
    if limit:
        images = images[:limit]
    print(f"[hailo_infer] {model}: {len(images)} images, hef={Path(hef_path).name}, "
          f"head={head} ({n_out} outputs)")

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
            if head == "nms":
                pipeline.set_nms_score_threshold(score_th)
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
                    if head == "nms":
                        raw = results[next(iter(results))][0]  # per-class list
                        preds.extend(_decode_nms(raw, imgsz, scale, pad_x, pad_y,
                                                 w0, h0, class_map, image_id, score_th))
                    elif head == "nmsfree":  # yolo26
                        preds.extend(_decode_nmsfree(results, imgsz, scale, pad_x,
                                                     pad_y, w0, h0, class_map,
                                                     image_id, score_th))
                    else:  # segment (yolov8-seg)
                        lb = LetterboxInfo(scale, pad_x, pad_y, w0, h0)
                        det, masks = _decode_seg(results, imgsz, lb, score_th)
                        preds.extend(detections_to_coco(image_id, det, class_map, masks))
                    t3 = time.perf_counter()
                    t_pre += t1 - t0
                    t_inf += t2 - t1
                    t_post += t3 - t2
                    if (i + 1) % 500 == 0:
                        print(f"  {i + 1}/{len(images)} ({len(preds)} dets)")
        wall = time.perf_counter() - wall0

    n = len(images)
    seg = head == "segment"
    iou_types = ("bbox", "segm") if seg else ("bbox",)
    print(f"[hailo_infer] scoring {len(preds)} detections over {n} images…")
    metrics = canonical_eval(str(gt), preds, iou_types=iou_types)

    cfg = {
        "bbox": metrics["bbox"],
        "segm": metrics.get("segm"),
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
        "label": model, "task": "segment" if seg else "detect",
        "host": {"machine": platform.machine(), "node": platform.node(),
                 "system": platform.platform(), "device": "rpi5-hailo8l"},
        "configs": {"yv-hailo": cfg},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"benchmark_a_{model}_{ts}.json"
    out_path.write_text(json.dumps(doc, indent=2))
    b = metrics["bbox"]
    mask = f" mask AP={metrics['segm']['AP']:.4f}" if seg and metrics.get("segm") else ""
    print(f"[hailo_infer] {model}: box AP={b['AP']:.4f} AP50={b['AP50']:.4f}{mask} "
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
    ap.add_argument("--head", default="auto",
                    choices=["auto", "nms", "nmsfree", "segment"],
                    help="output decoder; auto picks by output-vstream count")
    a = ap.parse_args()
    run(a.hef, a.model, a.coco_val, a.gt, a.out, a.imgsz, a.limit, a.score_th,
        head=a.head)


if __name__ == "__main__":
    main()
