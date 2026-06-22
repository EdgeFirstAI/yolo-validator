# yolo_validator/pipeline.py
"""Serial, per-stage-timed validation pipeline.

One frame at a time: load/decode -> preprocess -> inference -> decode ->
mask. Each stage timed with perf_counter. A measurement-excluded warmup
runs real images through the full pipeline first. Per-image results feed
COCO output/eval; per-stage samples feed _stats.stage_stats.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ._stats import StageStats, stage_stats
from .backends import Backend
from .detections import Detections
from .imageio import decode_image
from .masks import materialize_masks_numpy, materialize_masks_torch
from .postprocess import make_postprocessor
from .preprocess import make_preprocessor

STAGES = ("load_decode", "preprocess", "inference", "decode", "mask")


def _best_mask_device(provider: str):
    """Best torch device for fast-mode mask materialization, or None if torch is
    unavailable (then the numpy path is used). Prefers an on-device accelerator:
    cuda when the ONNX backend is already on cuda, else mps on Apple Silicon, else
    cpu. Even torch-cpu F.interpolate is ~4x the per-detection numpy loop, and the
    torch path is bit-identical to numpy (tests/test_masks.py)."""
    try:
        import torch
    except ImportError:
        return None
    if provider == "cuda" and torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class FrameResult:
    image_path: str
    detections: Detections
    masks: list = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class RunStats:
    stages: dict[str, StageStats]
    n_images: int
    # Wall-clock seconds for the MEASURED loop only (warmup excluded), so
    # wall-clock FPS == n_images / wall_s is apples-to-apples vs Ultralytics.
    wall_s: float = 0.0
    # Inference batch size used. >1 means the per-stage 'inference' figure is
    # AMORTIZED (batch_time / batch_size), a throughput number, NOT batch=1
    # latency. fps_wall (n_images/wall_s) stays directly comparable across sizes.
    batch_size: int = 1


class ValidationPipeline:
    def __init__(self, backend: Backend, preprocess_path="auto", postprocess_path="auto",
                 conf=0.001, iou=0.7, max_det=300, with_masks=True):
        self.backend = backend
        self.spec = backend.spec
        self.with_masks = with_masks and self.spec.task == "segment"
        self.pre = make_preprocessor(preprocess_path, self.spec.input_w, self.spec.input_h)
        self.post = make_postprocessor(postprocess_path, self.spec, conf, iou, max_det)

        # Mask materializer: always the fastest available, chosen automatically
        # (no user flag). With PyTorch importable, the bit-identical torch path runs
        # on the best device — mps on Apple Silicon, cuda on NVIDIA, else torch-cpu
        # (all 4-5x the numpy loop, mirroring Ultralytics' process_mask_native). On a
        # torch-free install it falls back to the numpy path (which beats OpenCV's
        # channel-batched decode and keeps the no-PyTorch edge guarantee).
        # _mask_device=None => numpy.
        self._mask_device = _best_mask_device(getattr(backend, "provider", "cpu"))

    def infer_one(self, image_path: str) -> FrameResult:
        timings: dict[str, float] = {}

        t = time.perf_counter()
        img, w, h = decode_image(image_path)
        timings["load_decode"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        tensor, lb = self.pre.preprocess(img, w, h)
        timings["preprocess"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        raw, dev = self.backend.run(tensor)
        timings["inference"] = (time.perf_counter() - t) * 1e3
        if dev is not None:
            timings["dma_input"] = dev.dma_input_ms
            timings["npu_compute"] = dev.compute_ms
            timings["dma_output"] = dev.dma_output_ms

        t = time.perf_counter()
        det = self.post.decode(raw, lb)
        timings["decode"] = (time.perf_counter() - t) * 1e3

        masks = []
        t = time.perf_counter()
        if self.with_masks and det.protos is not None and len(det.scores):
            if self._mask_device is not None:
                masks = materialize_masks_torch(
                    det.protos, det.coeffs, det.boxes_lb, lb,
                    self.spec.input_w, self.spec.input_h,
                    device=self._mask_device,
                )
            else:
                masks = materialize_masks_numpy(
                    det.protos, det.coeffs, det.boxes_lb, lb,
                    self.spec.input_w, self.spec.input_h,
                )
        timings["mask"] = (time.perf_counter() - t) * 1e3

        return FrameResult(image_path, det, masks, timings)

    def _infer_batch(self, image_paths_chunk: list[str]) -> list[FrameResult]:
        """Process B images in ONE backend call (batch_size > 1).

        Decode + preprocess are per-image (then stacked into (B,3,H,W)); inference
        is a single batched call whose time is AMORTIZED per-image (batch_time / B,
        a throughput figure — not a latency claim); decode + mask stay per-image, so
        a bounded batch keeps peak memory reasonable. ``on_frame`` is still invoked
        once per image by ``run``, preserving the streaming RLE encode within the
        batch window. Requires a dynamic-batch ONNX (see export_to_onnx); fixed
        batch=1 models (native CoreML / TensorRT) are rejected in ``run``.
        """
        b = len(image_paths_chunk)
        per: list[dict[str, float]] = [{} for _ in range(b)]
        tensors, lbs = [], []
        for j, p in enumerate(image_paths_chunk):
            t = time.perf_counter()
            img, w, h = decode_image(p)
            per[j]["load_decode"] = (time.perf_counter() - t) * 1e3
            t = time.perf_counter()
            tensor, lb = self.pre.preprocess(img, w, h)
            per[j]["preprocess"] = (time.perf_counter() - t) * 1e3
            tensors.append(tensor)
            lbs.append(lb)

        batched = np.concatenate(tensors, axis=0)        # (B, 3, H, W)
        t = time.perf_counter()
        raw, dev = self.backend.run(batched)
        inf_ms = (time.perf_counter() - t) * 1e3
        # Amortize the single batched inference across B frames -> per-image
        # throughput figure, comparable to batch=1. NOT a per-frame latency.
        for j in range(b):
            per[j]["inference"] = inf_ms / b
            if dev is not None:
                per[j]["dma_input"] = dev.dma_input_ms / b
                per[j]["npu_compute"] = dev.compute_ms / b
                per[j]["dma_output"] = dev.dma_output_ms / b

        results: list[FrameResult] = []
        for j, p in enumerate(image_paths_chunk):
            raw_j = [o[j:j + 1] for o in raw]            # per-image slice, keeps batch=1
            t = time.perf_counter()
            det = self.post.decode(raw_j, lbs[j])
            per[j]["decode"] = (time.perf_counter() - t) * 1e3
            masks = []
            t = time.perf_counter()
            if self.with_masks and det.protos is not None and len(det.scores):
                if self._mask_device is not None:
                    masks = materialize_masks_torch(
                        det.protos, det.coeffs, det.boxes_lb, lbs[j],
                        self.spec.input_w, self.spec.input_h, device=self._mask_device,
                    )
                else:
                    masks = materialize_masks_numpy(
                        det.protos, det.coeffs, det.boxes_lb, lbs[j],
                        self.spec.input_w, self.spec.input_h,
                    )
            per[j]["mask"] = (time.perf_counter() - t) * 1e3
            results.append(FrameResult(p, det, masks, per[j]))
        return results

    def run(self, image_paths: list[str], warmup: int = 3,
            on_frame=None, warmup_on_frame=None, batch_size: int = 1) -> RunStats:
        if batch_size > 1:
            mb = getattr(self.backend, "input_batch", 1)
            if mb is not None and mb == 1:
                raise ValueError(
                    f"batch_size={batch_size} requested but the model has a fixed "
                    f"batch dimension of 1. Re-export ONNX with a dynamic batch axis "
                    f"(export_to_onnx dynamic=True) and use an ONNX EP. Native CoreML "
                    f"(.mlpackage) and TensorRT engines cannot dynamic-batch."
                )

        for p in image_paths[: max(0, warmup)]:
            res = self.infer_one(p)  # warmup: excluded from timing
            if warmup_on_frame is not None:
                warmup_on_frame(res)

        samples: dict[str, list[float]] = {s: [] for s in STAGES}
        measured = image_paths[max(0, warmup):]

        def _collect(res):
            for s in STAGES:
                if s in res.timings:
                    samples[s].append(res.timings[s])
            if on_frame is not None:
                on_frame(res)

        t_measured = time.perf_counter()
        if batch_size <= 1:
            for p in measured:
                _collect(self.infer_one(p))
        else:
            for i in range(0, len(measured), batch_size):
                for res in self._infer_batch(measured[i:i + batch_size]):
                    _collect(res)
        wall_s = time.perf_counter() - t_measured
        stages = {s: stage_stats(v) for s, v in samples.items() if v}
        return RunStats(stages=stages, n_images=len(measured), wall_s=wall_s,
                        batch_size=batch_size)
