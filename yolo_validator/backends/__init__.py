"""Backend abstraction.

A backend loads a model and runs one frame. `run()` returns the raw model
outputs plus optional device sub-timing (DMA/compute) when the runtime
exposes it; None otherwise. New runtimes (TensorRT, TFLite, HailoRT, Ara2)
implement this protocol in their own modules and register in load_backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    input_w: int
    input_h: int
    task: str  # "detect" | "segment"
    e2e: bool = False


def infer_e2e(output_shapes, override: Optional[bool] = None) -> bool:
    """Decide whether a model is NMS-free / end-to-end from its output shapes.

    ``override`` (from ``--e2e/--no-e2e`` or model metadata) wins when given.

    Otherwise: an E2E head emits ``[1, num_det, 6(+nm)]`` — few rows
    (num_det, typically 300) and small columns (``6`` or ``6+nm``). A classic
    anchor-grid head is ``[1, channels, anchors]`` (or transposed
    ``[1, anchors, channels]``) where the anchor dimension is in the
    thousands. Requiring BOTH dims small avoids mis-routing a transposed
    classic output (e.g. ``[1, 8400, 84]`` or a low-class ``[1, 8400, 5]``)
    to the E2E decoder. If a non-standard export defeats this heuristic, pass
    an explicit override.
    """
    if override is not None:
        return override
    for s in output_shapes:
        if len(s) == 3 and isinstance(s[1], int) and isinstance(s[2], int):
            if s[2] <= 64 and s[1] <= 1024:
                return True
    return False


@dataclass(frozen=True)
class DeviceTiming:
    dma_input_ms: float
    compute_ms: float
    dma_output_ms: float

    @property
    def total_ms(self) -> float:
        return self.dma_input_ms + self.compute_ms + self.dma_output_ms


@runtime_checkable
class Backend(Protocol):
    spec: ModelSpec

    def run(self, input_tensor: np.ndarray) -> tuple[list[np.ndarray], Optional[DeviceTiming]]:
        ...


def load_backend(model_path: str, runtime: str = "onnx", **opts) -> Backend:
    """Construct a backend by runtime name ('onnx' | 'tensorrt')."""
    if runtime == "onnx":
        from .onnxruntime_backend import OnnxRuntimeBackend

        return OnnxRuntimeBackend(model_path, **opts)
    if runtime == "tensorrt":
        from .tensorrt_backend import TensorRTBackend

        # The ONNX-only 'provider' opt does not apply to TensorRT.
        opts.pop("provider", None)
        return TensorRTBackend(model_path, **opts)
    raise ValueError(f"unsupported runtime: {runtime!r} (expected 'onnx' or 'tensorrt')")
