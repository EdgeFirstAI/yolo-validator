"""On-device Ara240 (Kinara Ara-2 NPU) dvapi inference + COCO scoring.

Reference lane for ``imx95-ara240``: vendor pre-compiled ``.dvm`` models run
through the vendor host stack — OpenCV letterbox → int8 quantize → dvapi
``infer_sync`` (ctypes over libaraclient via the dvproxy Unix socket) →
dequantize → the same shared NumPy decode the other lanes use.

Because libaraclient aborts (glibc double free) after roughly 100–200
sequential inferences in one process, the orchestrator runs the val set in
**chunked subprocesses** (``--chunk``, default 100, retried once), merges the
partial predictions/timings, and scores once with ``canonical_eval``
(crowd-as-normal). Per frame it records the wall-clock inference roundtrip
plus the driver-reported sub-timings: DMA host→device, **core NPU compute**,
DMA device→host (dvapi reports them in µs despite its struct comments saying
ms). ``pre`` = letterbox+quantize (JPEG decode is reported separately under
``stats`` only); ``post`` = dequant+NMS+masks+COCO/RLE (same as the tflite
lane); ``e2e`` = pre+inf+post; single-stream, so stages are additive.

Emits ``benchmark_a_<model>_<ts>.json`` (config key ``yv-ara2``) that
``benchmarks.normalize`` folds into ``metrics/imx95-ara240.json``. The
``--workflow-vendor`` label records artifact provenance per row
(``kinara-sdk-1.2.1`` local DVMs vs ``nxp-hf-2.0.4`` from HF ``nxp/YOLOv8``).

Usage (on the target; scoring can be deferred to the host with --no-eval)::

    python3 -m benchmarks.ara2_infer --dvm yolov8n-seg-kinara-1.2.1.dvm \
        --model yolov8n-seg --workflow-vendor kinara-sdk-1.2.1 \
        --coco-val ~/coco/val2017 \
        --gt ~/coco/annotations/instances_val2017.json --no-eval

    # host: score a merged file produced by --no-eval
    python -m benchmarks.ara2_infer --finalize merged_yolov8n-seg_x.json \
        --gt ~/coco/annotations/instances_val2017.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from yolo_validator.coco_output import coco80_to_coco91, detections_to_coco
from yolo_validator.detections import Detections
from yolo_validator.letterbox import LetterboxInfo, unletterbox_boxes
from yolo_validator.masks import materialize_masks_numpy
from yolo_validator.nms import nms_class_aware

_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# Per-frame stages (ms). npu_* are dvproxy driver stats (µs→ms), NaN when the
# driver returns no stats for a frame.
STAGE_KEYS = ("decode", "preprocess", "inference", "postprocess", "e2e",
              "npu_h2d", "npu_core", "npu_d2h")


def _safe_label(name: str) -> str:
    """Reject model labels with path/shell metacharacters before they reach a
    filename (defense-in-depth for untrusted CLI input)."""
    if not _LABEL_RE.fullmatch(name) or ".." in name:
        raise SystemExit(f"invalid model label {name!r}: expected [A-Za-z0-9._-]")
    return name


def _stats(vals) -> dict | None:
    """min/mean/p50/p95/p99/max/n over NaN-filtered vals — NO trimming."""
    a = np.asarray([v for v in vals if not math.isnan(v)], dtype=np.float64)
    if a.size == 0:
        return None
    return {"min": float(a.min()), "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()), "n": int(a.size)}


def _mean(vals) -> float | None:
    s = _stats(vals)
    return s["mean"] if s else None


def _strip_trailing_ones(shape: list[int]) -> list[int]:
    """Strip trailing size-1 dims (Ara240 shape padding convention)."""
    shape = list(shape)
    while len(shape) > 2 and shape[-1] == 1:
        shape = shape[:-1]
    return shape


def _infer_role(shape, seen: set) -> str:
    """Guess an output tensor's role from its (channels-first) shape."""
    s = _strip_trailing_ones(list(shape))
    if len(s) == 3 and 32 in s and max(s) > 100:
        return "protos"
    if len(s) >= 2:
        feat = min(s[0], s[-1]) if len(s) == 2 else s[0]
        if feat == 4:
            return "boxes"
        if feat == 2:  # split decoder: first (2,N) is box_xy, second box_wh
            return "box_wh" if "box_xy" in seen else "box_xy"
        if feat == 80:
            return "scores"
        if feat == 32:
            return "mask_coefs"
    return "unknown"


def _to_nc(a: np.ndarray) -> np.ndarray:
    """(1,C,N) | (C,N) | (N,C) → (N,C), assuming C < N."""
    if a.ndim == 3:
        a = a[0]
    return a.T if a.shape[0] < a.shape[1] else a


def _decode_outputs(outputs, imgsz, lb, image_id, class_map, score_th,
                    iou=0.7, max_det=300, top_k=1000):
    """Dequantized role tensors → COCO prediction dicts (original coords).

    outputs: "boxes" (4,N) xywh letterbox px — or "box_xy"+"box_wh" (2,N)
    each — plus "scores" (nc,N) in-graph-sigmoid'd probs, and for seg
    "mask_coefs" (32,N) + "protos" (32,ph,pw).
    """
    if "boxes" in outputs:
        boxes = _to_nc(outputs["boxes"])                      # (N,4) xywh
    else:
        boxes = np.concatenate(
            [_to_nc(outputs["box_xy"]), _to_nc(outputs["box_wh"])], axis=1)
    # Orient scores explicitly by expected nc, not via generic heuristic
    nc = len(class_map)
    raw = outputs["scores"]
    raw = raw[0] if raw.ndim == 3 else raw
    scores = raw.T if raw.shape[0] == nc else raw              # (N,nc)
    cx, cy, w, h = boxes.T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    flat = scores.reshape(-1)
    keep = np.nonzero(flat > score_th)[0]
    if not keep.size:
        return []
    if keep.size > top_k:
        keep = keep[np.argpartition(flat[keep], -top_k)[-top_k:]]
    anc, cl, sc = keep // nc, keep % nc, flat[keep]
    box_lb = xyxy[anc]
    idx = nms_class_aware(box_lb, sc, cl, iou)[:max_det]
    box_lb, cls, sc = box_lb[idx], cl[idx].astype(np.int64), sc[idx]
    if "protos" in outputs and "mask_coefs" in outputs:
        coeffs = _to_nc(outputs["mask_coefs"])[anc][idx]
        proto = outputs["protos"]
        if proto.ndim == 4:
            proto = proto[0]                                  # (32,ph,pw)
        det = Detections(boxes=unletterbox_boxes(box_lb, lb), scores=sc,
                         classes=cls, coeffs=coeffs,
                         protos=proto[None].astype(np.float32),
                         boxes_lb=box_lb)
        masks = materialize_masks_numpy(det.protos, det.coeffs, det.boxes_lb,
                                        lb, imgsz, imgsz)
        return detections_to_coco(image_id, det, class_map, masks)
    det = Detections(boxes=unletterbox_boxes(box_lb, lb), scores=sc,
                     classes=cls)
    return detections_to_coco(image_id, det, class_map, None)


def _list_images(coco_val, limit: int) -> list[Path]:
    images = sorted(Path(coco_val).expanduser().glob("*.jpg"))
    return images[:limit] if limit else images


def _merge_partials(paths) -> dict:
    parts = sorted((json.loads(Path(p).read_text()) for p in paths),
                   key=lambda d: d["start"])
    if not parts:
        raise SystemExit("no partial results to merge")
    merged = {"preds": [], "timings": {k: [] for k in STAGE_KEYS},
              "wall_s": 0.0, "n_images": 0, "imgsz": parts[0]["imgsz"]}
    for d in parts:
        merged["preds"].extend(d["preds"])
        merged["wall_s"] += d["wall_s"]
        merged["n_images"] += d["count"]
        for k in STAGE_KEYS:
            merged["timings"][k].extend(d["timings"].get(k, []))
    return merged


def _build_doc(merged: dict, metrics: dict | None) -> dict:
    meta, tm = merged["meta"], merged["timings"]
    cfg = {
        "bbox": (metrics or {}).get("bbox"),
        "segm": (metrics or {}).get("segm"),
        "timing": {"preprocess": _mean(tm["preprocess"]),
                   "inference": _mean(tm["inference"]),
                   "postprocess": _mean(tm["postprocess"]),
                   "e2e": _mean(tm["e2e"])},
        "npu": {"h2d": _mean(tm["npu_h2d"]), "core": _mean(tm["npu_core"]),
                "d2h": _mean(tm["npu_d2h"])},
        "stats": {k: _stats(tm[k]) for k in STAGE_KEYS},
        "fps_wall": (merged["n_images"] / merged["wall_s"]
                     if merged["wall_s"] else None),
        "n_images": merged["n_images"], "batch": 1,
        "vendor": meta["vendor"], "artifact": meta["artifact"],
        "chunks": meta["chunks"], "chunk_retries": meta["chunk_retries"],
    }
    return {"label": meta["model"], "task": meta["task"],
            "host": meta["host"], "configs": {"yv-ara2": cfg}}


def finalize(merged_path, gt, out_dir) -> Path:
    """Score a merged predictions file and emit the benchmark_a document.

    Runs anywhere with pycocotools — lets the target run --no-eval and the
    host do the scoring.
    """
    merged = json.loads(Path(merged_path).read_text())
    meta = merged["meta"]
    from benchmarks.canonical_eval import canonical_eval
    iou_types = ("bbox", "segm") if meta["task"] == "segment" else ("bbox",)
    print(f"[ara2_infer] scoring {len(merged['preds'])} dets over "
          f"{merged['n_images']} images…")
    metrics = canonical_eval(str(Path(gt).expanduser()), merged["preds"],
                             iou_types=iou_types)
    doc = _build_doc(merged, metrics)
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"benchmark_a_{meta['model']}_{ts}.json"
    out_path.write_text(json.dumps(doc, indent=2))
    cfg, b = doc["configs"]["yv-ara2"], metrics["bbox"]
    mask = (f" mask AP={metrics['segm']['AP']:.4f}"
            if metrics.get("segm") else "")
    print(f"[ara2_infer] {meta['model']} [{meta['vendor']}]: "
          f"box AP={b['AP']:.4f} AP50={b['AP50']:.4f}{mask} | "
          f"{cfg['fps_wall']:.2f} fps | inf {cfg['timing']['inference']:.1f} ms "
          f"(npu core {cfg['npu']['core'] or float('nan'):.2f} ms) -> {out_path}")
    return out_path


# ── On-target worker (dvapi imported lazily — hosts have no libaraclient) ──

def _parse_model(dv_model) -> dict:
    """Input/output metadata from a loaded DVModel (dvapi structs)."""
    ip = dv_model.input_param[0]
    pp = ip.preprocess_param
    meta = {"h": ip.height, "w": ip.width, "c": ip.nch,
            "qn": pp.qn, "offset": pp.offset, "signed": bool(pp.is_signed),
            "outputs": {}}
    seen: set = set()
    for j in range(dv_model.num_outputs):
        op = dv_model.output_param[j]
        shape = _strip_trailing_ones([op.nch, op.height, op.width])
        if op.depth > 1:
            shape = _strip_trailing_ones([op.nch, op.depth,
                                          op.height, op.width])
        role = _infer_role(shape, seen)
        seen.add(role)
        opp = op.postprocess_param
        if op.bpp == 1:
            dtype = "int8" if opp.is_signed else "uint8"
        elif op.bpp == 2:
            dtype = "int16" if opp.is_signed else "uint16"
        else:
            dtype = "float32"
        meta["outputs"][role] = {"index": j, "shape": tuple(shape),
                                 "dtype": dtype, "qn": opp.qn,
                                 "offset": opp.offset}
    return meta


def _preprocess(bgr, h, w, qn, offset, signed):
    """BGR → letterbox → [0,1] → affine int8 quantize → CHW flat.

    Matches Ultralytics LetterBox(center=True) + the Ara240 input quant
    (quantized = round(pixel/255 / qn + offset)).
    """
    h0, w0 = bgr.shape[:2]
    scale = min(w / w0, h / h0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    pad_x, pad_y = (w - nw) // 2, (h - nh) // 2
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(resized, pad_y, h - nh - pad_y,
                                pad_x, w - nw - pad_x,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    q = np.round(x / qn + offset)
    dt = np.int8 if signed else np.uint8
    info = np.iinfo(dt)
    q = np.clip(q, info.min, info.max).astype(dt)
    flat = np.ascontiguousarray(q.transpose(2, 0, 1)).reshape(-1)
    return flat, LetterboxInfo(scale, pad_x, pad_y, w0, h0)


def worker(a) -> None:
    """Process images[start:start+count] in THIS process and write a partial.

    Kept to ≤ --chunk images because libaraclient double-frees after
    ~100–200 sequential inferences; the orchestrator restarts us per chunk.
    """
    from benchmarks import ara2_dvapi as dvapi
    images = _list_images(a.coco_val, a.limit)
    chunk = images[a.start:a.start + a.count]
    ret, session = dvapi.DVSession.create_via_unix_socket(a.socket)
    if ret != dvapi.dv_status_code.DV_SUCCESS:
        raise SystemExit(f"cannot connect to dvproxy at {a.socket}: {ret} "
                         "(start it: systemctl start dvproxy)")
    with session:
        ret, endpoints = session.get_endpoint_list()
        if ret != dvapi.dv_status_code.DV_SUCCESS or not endpoints:
            raise SystemExit("no Ara240 endpoints available")
        ret, model = session.load_model_from_file(endpoints[0], str(a.dvm))
        if ret != dvapi.dv_status_code.DV_SUCCESS:
            raise SystemExit(f"model load failed for {a.dvm}: {ret}")
        meta = _parse_model(model)
        out_tensors = model._allocate_output_tensors()
        class_map = coco80_to_coco91()
        imgsz = meta["w"]
        if a.start == 0:
            print(f"[worker] input {meta['c']}x{meta['h']}x{meta['w']} "
                  f"qn={meta['qn']:.6f} offset={meta['offset']}; outputs: "
                  + ", ".join(f"{r}={m['shape']}/{m['dtype']}"
                              for r, m in meta["outputs"].items()))
            if "unknown" in meta["outputs"] or ("boxes" not in meta["outputs"] \
                    and "box_xy" not in meta["outputs"]):
                raise SystemExit(f"unrecognized output layout: "
                                 f"{[(r, m['shape']) for r, m in meta['outputs'].items()]}")

        def infer_one(path, timings=None, preds=None):
            t_start = time.perf_counter()
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise RuntimeError(f"failed to decode {path}")
            t0 = time.perf_counter()
            flat, lb = _preprocess(bgr, meta["h"], meta["w"], meta["qn"],
                                   meta["offset"], meta["signed"])
            t1 = time.perf_counter()
            tensor = dvapi.DVTensor(flat, model.input_param[0])
            ret, req = model.infer_sync([tensor], out_tensors)
            t2 = time.perf_counter()
            if ret != dvapi.dv_status_code.DV_SUCCESS:
                raise RuntimeError(f"inference failed on {path.name}: {ret}")
            stats = req.stats
            outputs = {}
            for role, om in meta["outputs"].items():
                raw = out_tensors[om["index"]].numpy_data.copy()
                dt = np.dtype(om["dtype"])
                if dt != np.int8:
                    raw = raw.view(dt)
                raw = raw.reshape(om["shape"])
                outputs[role] = (raw.astype(np.float32) - om["offset"]) * om["qn"]
            recs = _decode_outputs(outputs, imgsz, lb, int(path.stem),
                                   class_map, a.score_th, a.iou, a.max_det)
            t3 = time.perf_counter()
            if timings is not None:
                timings["decode"].append((t0 - t_start) * 1e3)
                timings["preprocess"].append((t1 - t0) * 1e3)
                timings["inference"].append((t2 - t1) * 1e3)
                timings["postprocess"].append((t3 - t2) * 1e3)
                timings["e2e"].append((t3 - t0) * 1e3)
                # dvapi struct comments say ms; firmware actually reports µs.
                timings["npu_h2d"].append(
                    stats.input_transfer_time / 1000 if stats else float("nan"))
                timings["npu_core"].append(
                    stats.npu_compute_ms if stats else float("nan"))
                timings["npu_d2h"].append(
                    stats.output_transfer_time / 1000 if stats else float("nan"))
            if preds is not None:
                preds.extend(recs)

        for p in chunk[:min(a.warmup, len(chunk))]:
            infer_one(p)                       # warmup, unmeasured
        timings = {k: [] for k in STAGE_KEYS}
        preds: list[dict] = []
        wall0 = time.perf_counter()
        for p in chunk:
            infer_one(p, timings, preds)
        wall = time.perf_counter() - wall0
        # Explicit unload before session __exit__: DVModel.__del__ after
        # disconnect double-frees (see ara2-validator reference.py).
        model.unload()
    partial = {"start": a.start, "count": len(chunk), "wall_s": wall,
               "imgsz": imgsz, "preds": preds, "timings": timings}
    a.partial_out.parent.mkdir(parents=True, exist_ok=True)
    a.partial_out.write_text(json.dumps(partial))
    print(f"[worker] chunk {a.start}+{len(chunk)}: {len(preds)} dets, "
          f"{wall:.1f}s -> {a.partial_out}")


# ── Orchestrator (parent process) ──────────────────────────────────────────

def orchestrate(a) -> None:
    images = _list_images(a.coco_val, a.limit)
    if not images:
        raise SystemExit(f"no *.jpg under {a.coco_val}")
    n = len(images)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(a.out).expanduser()
    partial_dir = out_dir / f"partials_{a.model}_{ts}"
    starts = list(range(0, n, a.chunk))
    print(f"[ara2_infer] {a.model} [{a.workflow_vendor}] {Path(a.dvm).name}: "
          f"{n} images in {len(starts)} chunks of {a.chunk}")
    retries = 0
    for start in starts:
        count = min(a.chunk, n - start)
        partial = partial_dir / f"partial_{start:06d}.json"
        cmd = [sys.executable, "-m", "benchmarks.ara2_infer", "--worker",
               "--dvm", str(a.dvm), "--model", a.model,
               "--coco-val", str(a.coco_val), "--limit", str(a.limit),
               "--start", str(start), "--count", str(count),
               "--partial-out", str(partial), "--socket", a.socket,
               "--warmup", str(a.warmup), "--score-th", str(a.score_th),
               "--iou", str(a.iou), "--max-det", str(a.max_det)]
        for attempt in (1, 2):
            if subprocess.call(cmd) == 0:
                break
            retries += 1
            print(f"[ara2_infer] chunk {start} failed "
                  f"(attempt {attempt}/2, first image {images[start].name})")
        else:
            raise SystemExit(f"chunk {start} failed twice; aborting "
                             f"(first image: {images[start].name})")
    merged = _merge_partials(sorted(partial_dir.glob("partial_*.json")))
    merged["meta"] = {
        "model": a.model,
        "task": "segment" if "seg" in a.model else "detect",
        "vendor": a.workflow_vendor, "artifact": Path(a.dvm).name,
        "device": a.device_label, "chunks": len(starts),
        "chunk_retries": retries,
        "host": {"machine": platform.machine(), "node": platform.node(),
                 "system": platform.platform(), "device": a.device_label}}
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir / f"merged_{a.model}_{a.workflow_vendor}_{ts}.json"
    merged_path.write_text(json.dumps(merged))
    print(f"[ara2_infer] merged {merged['n_images']} images, "
          f"{len(merged['preds'])} dets -> {merged_path}")
    if a.no_eval:
        print(f"[ara2_infer] --no-eval: score later with\n  python -m "
              f"benchmarks.ara2_infer --finalize {merged_path} --gt <gt.json>")
        return
    finalize(merged_path, a.gt, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dvm", type=Path, help="pre-compiled .dvm model")
    ap.add_argument("--model", help="variant label, e.g. yolov8n / yolov8n-seg")
    ap.add_argument("--workflow-vendor", default="kinara-sdk-1.2.1",
                    help="artifact provenance recorded per row "
                         "(kinara-sdk-1.2.1 | nxp-hf-2.0.4)")
    ap.add_argument("--device-label", default="imx95-ara240")
    ap.add_argument("--coco-val", type=Path,
                    default=Path.home() / "coco" / "val2017")
    ap.add_argument("--gt", type=Path, default=Path.home() / "coco"
                    / "annotations" / "instances_val2017.json")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results/ara2"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all val2017")
    ap.add_argument("--chunk", type=int, default=100,
                    help="images per worker process (libaraclient crashes "
                         "past ~100-200 inferences per process)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--score-th", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--socket", default="/var/run/ara2.sock")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip scoring (score on the host via --finalize)")
    ap.add_argument("--finalize", type=Path,
                    help="score a merged_*.json and emit the benchmark doc")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--start", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--count", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--partial-out", type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.finalize:
        finalize(a.finalize, a.gt, a.out)
        return
    if not a.dvm or not a.model:
        ap.error("--dvm and --model are required")
    a.model = _safe_label(a.model)
    if not a.dvm.expanduser().exists():
        raise SystemExit(f"not found: {a.dvm}")
    a.dvm = a.dvm.expanduser().resolve()
    if a.worker:
        worker(a)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
