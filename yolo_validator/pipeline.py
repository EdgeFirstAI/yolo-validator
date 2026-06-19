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

from ._stats import StageStats, stage_stats
from .backends import Backend
from .detections import Detections
from .imageio import decode_image
from .masks import materialize_masks_numpy, materialize_masks_torch
from .postprocess import make_postprocessor
from .preprocess import make_preprocessor

STAGES = ("load_decode", "preprocess", "inference", "decode", "mask")


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


class ValidationPipeline:
    def __init__(self, backend: Backend, preprocess_path="auto", postprocess_path="auto",
                 conf=0.001, iou=0.7, max_det=300, with_masks=True):
        self.backend = backend
        self.spec = backend.spec
        self.with_masks = with_masks and self.spec.task == "segment"
        self.pre = make_preprocessor(preprocess_path, self.spec.input_w, self.spec.input_h)
        self.post = make_postprocessor(postprocess_path, self.spec, conf, iou, max_det)

        # Choose mask materializer: torch (GPU-capable) when postprocess_path is
        # "torch" and the backend is using CUDAExecutionProvider; numpy otherwise.
        # The yv-numpy path is always CPU regardless of provider (edge-portable).
        _backend_provider = getattr(backend, "provider", "cpu")
        if postprocess_path == "torch" and _backend_provider == "cuda":
            self._mask_device = "cuda"
        else:
            self._mask_device = "cpu"

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
            if self._mask_device == "cuda":
                masks = materialize_masks_torch(
                    det.protos, det.coeffs, det.boxes_lb, lb,
                    self.spec.input_w, self.spec.input_h,
                    device="cuda",
                )
            else:
                masks = materialize_masks_numpy(
                    det.protos, det.coeffs, det.boxes_lb, lb,
                    self.spec.input_w, self.spec.input_h,
                )
        timings["mask"] = (time.perf_counter() - t) * 1e3

        return FrameResult(image_path, det, masks, timings)

    def run(self, image_paths: list[str], warmup: int = 3,
            on_frame=None, warmup_on_frame=None) -> RunStats:
        for p in image_paths[: max(0, warmup)]:
            res = self.infer_one(p)  # warmup: excluded from timing
            if warmup_on_frame is not None:
                warmup_on_frame(res)

        samples: dict[str, list[float]] = {s: [] for s in STAGES}
        measured = image_paths[max(0, warmup):]
        t_measured = time.perf_counter()
        for p in measured:
            res = self.infer_one(p)
            for s in STAGES:
                if s in res.timings:
                    samples[s].append(res.timings[s])
            if on_frame is not None:
                on_frame(res)
        wall_s = time.perf_counter() - t_measured
        stages = {s: stage_stats(v) for s, v in samples.items() if v}
        return RunStats(stages=stages, n_images=len(measured), wall_s=wall_s)
