"""ONNX Runtime backend (CPU/CUDA/CoreML execution providers).

Host-only: no device DMA sub-timing, so run() returns None for timing.
Task (detect vs segment) is inferred from output count unless given.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import DeviceTiming, ModelSpec, infer_e2e

_PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}


class OnnxRuntimeBackend:
    def __init__(self, model_path: str, provider: str = "cpu", task: Optional[str] = None,
                 e2e: Optional[bool] = None):
        import onnxruntime as ort

        self.provider = provider  # "cpu" | "cuda" | "coreml" — exposed for pipeline device selection
        self.session = ort.InferenceSession(model_path, providers=_PROVIDERS[provider])
        inp = self.session.get_inputs()[0]
        shape = inp.shape  # [N, C, H, W]
        h = int(shape[2]) if isinstance(shape[2], int) else 640
        w = int(shape[3]) if isinstance(shape[3], int) else 640
        self.input_name = inp.name
        outputs = self.session.get_outputs()
        inferred = "segment" if len(outputs) >= 2 else "detect"
        is_e2e = infer_e2e([o.shape for o in outputs], override=e2e)

        self.spec = ModelSpec(input_w=w, input_h=h, task=task or inferred, e2e=is_e2e)

    def run(self, input_tensor: np.ndarray) -> tuple[list[np.ndarray], Optional[DeviceTiming]]:
        outs = self.session.run(None, {self.input_name: input_tensor.astype(np.float32)})
        return list(outs), None
