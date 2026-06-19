# benchmarks/rebin.py
"""Re-binning of pipeline stage timings into canonical {preprocess, inference, postprocess, e2e}.

Design rule (from design doc §5):
  preprocess  → preprocess   (direct)
  inference   → inference    (direct)
  postprocess → decode + mask + encode (summed per-image, then stats)
  e2e         → preprocess + inference + decode + mask + encode (summed per-image, then stats)
  load_decode is excluded from the canonical vector.

encode covers output serialisation: D2H for the torch GPU mask tensor and
RLE encoding via pycocotools. Including it in postprocess/e2e ensures
fps_pipeline is computed from the complete per-frame cost with no hidden stages.

All rebin functions operate on per-image raw samples (not aggregate stats)
because you cannot sum means — you must sum per-image values first, then aggregate.
"""
from __future__ import annotations

from yolo_validator._stats import StageStats, stage_stats


def rebin_frame(timings: dict[str, float]) -> dict[str, float]:
    """Map one frame's fine-stage timings to canonical {preprocess, inference, postprocess, e2e}.

    Accepts either raw pipeline stage keys (decode, mask) or already-canonical keys
    (postprocess). If 'postprocess' is present it is used directly; otherwise it is
    computed as decode + mask. Similarly, 'e2e' is used directly if present.

    Args:
        timings: dict with keys from pipeline STAGES (load_decode, preprocess, inference,
                 decode, mask) or already-canonical keys (preprocess, inference, postprocess,
                 e2e). Missing stages are treated as 0.

    Returns:
        dict with keys {preprocess, inference, postprocess, e2e} in ms.
        load_decode is excluded from the canonical vector.
    """
    pre = timings.get("preprocess", 0.0)
    inf = timings.get("inference", 0.0)
    enc = timings.get("encode", 0.0)   # output serialisation: D2H + RLE encode

    if "postprocess" in timings:
        post = timings["postprocess"] + enc
    else:
        dec = timings.get("decode", 0.0)
        msk = timings.get("mask", 0.0)
        post = dec + msk + enc

    if "e2e" in timings:
        e2e = timings["e2e"] + enc
    else:
        e2e = pre + inf + post

    out = {
        "preprocess": pre,
        "inference": inf,
        "postprocess": post,
        "e2e": e2e,
    }
    # Carry through device sub-timings when a backend exposes them (TensorRT
    # CUDA events): npu_compute is the on-device GPU compute time — the honest
    # value for an "inference (GPU)" column — vs the host-measured 'inference'
    # which also includes H2D/D2H copies and stream sync.
    for dk in ("dma_input", "npu_compute", "dma_output"):
        if dk in timings:
            out[dk] = timings[dk]
    return out


def rebin_samples(per_frame_timings: list[dict[str, float]]) -> dict[str, StageStats]:
    """Aggregate a list of per-frame canonical timings into StageStats per bucket.

    Args:
        per_frame_timings: list of dicts, each produced by rebin_frame (keys:
                           preprocess, inference, postprocess, e2e).

    Returns:
        dict mapping canonical stage name -> StageStats.
    """
    if not per_frame_timings:
        raise ValueError("rebin_samples requires at least one frame")

    rebinned_frames = [rebin_frame(f) for f in per_frame_timings]
    keys: list[str] = ["preprocess", "inference", "postprocess", "e2e"]
    for extra in ("dma_input", "npu_compute", "dma_output"):
        if any(extra in rf for rf in rebinned_frames):
            keys.append(extra)
    buckets: dict[str, list[float]] = {k: [] for k in keys}

    for rf in rebinned_frames:
        for k in keys:
            if k in rf:
                buckets[k].append(rf[k])

    return {k: stage_stats(v) for k, v in buckets.items() if v}


def rebin_ultralytics(speed: dict[str, float], n_images: int) -> dict[str, float]:
    """Convert Ultralytics speed dict (ms/image means) to canonical vector.

    Ultralytics speed keys are already 'preprocess', 'inference', 'postprocess'
    (mean ms per image). This function just computes e2e and returns canonical means.

    Args:
        speed: dict with keys 'preprocess', 'inference', 'postprocess' (mean ms/image).
        n_images: number of measured images (informational, not used in mean computation).

    Returns:
        dict with keys {preprocess, inference, postprocess, e2e} — all mean ms/image.
    """
    pre = speed.get("preprocess", 0.0)
    inf = speed.get("inference", 0.0)
    post = speed.get("postprocess", 0.0)
    e2e = pre + inf + post

    return {
        "preprocess": pre,
        "inference": inf,
        "postprocess": post,
        "e2e": e2e,
    }
