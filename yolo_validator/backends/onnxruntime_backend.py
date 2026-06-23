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
        # FP16 graphs (e.g. ONNX exported with half=True) want float16 input; the
        # preprocessed tensor is always float32, so feed the dtype the graph declares.
        self.input_dtype = np.float16 if "float16" in inp.type else np.float32
        outputs = self.session.get_outputs()
        inferred = "segment" if len(outputs) >= 2 else "detect"
        is_e2e = infer_e2e([o.shape for o in outputs], override=e2e)

        self.spec = ModelSpec(input_w=w, input_h=h, task=task or inferred, e2e=is_e2e)

    def run(self, input_tensor: np.ndarray) -> tuple[list[np.ndarray], Optional[DeviceTiming]]:
        outs = self.session.run(None, {self.input_name: input_tensor.astype(self.input_dtype)})
        # Upcast FP16 outputs so the downstream NumPy/torch decode + mask pipeline
        # (which assumes float32) is unchanged. No-op for FP32 graphs.
        outs = [o.astype(np.float32) if o.dtype == np.float16 else o for o in outs]
        return outs, None
