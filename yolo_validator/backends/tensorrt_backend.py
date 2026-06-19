"""TensorRT backend (Jetson / NVIDIA GPU).

Loads a serialized TensorRT engine and runs inference via the TRT 10.x
name-based I/O-tensor API with pycuda-managed device buffers. Produces the
same raw output arrays as the ONNX backend, so the rest of the validator
(decode / mask / eval) is unchanged.

The loader accepts both a raw serialized engine and an Ultralytics-exported
``.engine`` (which prepends a 4-byte little-endian length + a JSON metadata
header before the engine bytes). Stripping that header lets yolo-validator
load the *identical* engine Ultralytics runs, so the TensorRT equivalence
comparison shares one engine rather than two independently-built ones.

Per-phase device timing (H2D / compute / D2H) is captured with CUDA events
and returned as ``DeviceTiming`` — the same shape the Ara2 path exposes.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np

from . import DeviceTiming, ModelSpec, infer_e2e


def strip_ultralytics_metadata(data: bytes) -> bytes:
    """Return the raw engine bytes, stripping an Ultralytics metadata header.

    Ultralytics writes ``int32_le(len(meta)) + meta_json + engine``. If the
    leading 4 bytes describe a length whose following bytes parse as JSON,
    the remainder is the engine; otherwise ``data`` is already a raw engine.
    """
    if len(data) > 4:
        meta_len = int.from_bytes(data[:4], byteorder="little", signed=True)
        if 0 < meta_len < len(data) - 4:
            try:
                json.loads(data[4 : 4 + meta_len].decode("utf-8"))
                return data[4 + meta_len :]
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
    return data


def _trt_to_np(trt):
    return {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT32: np.int32,
        trt.DataType.INT8: np.int8,
        trt.DataType.BOOL: np.bool_,
    }


class TensorRTBackend:
    """Runs a TensorRT engine; output layout matches the ONNX backend."""

    def __init__(self, engine_path: str, task: Optional[str] = None,
                 e2e: Optional[bool] = None):
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401 — creates the CUDA context
        import pycuda.driver as cuda

        self._cuda = cuda
        npmap = _trt_to_np(trt)

        with open(engine_path, "rb") as f:
            engine_bytes = strip_ultralytics_metadata(f.read())
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(int(d) for d in self.engine.get_tensor_shape(name))
            dtype = npmap[self.engine.get_tensor_dtype(name)]
            count = int(np.prod(shape))
            entry = {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "host": cuda.pagelocked_empty(count, dtype),
                "dev": cuda.mem_alloc(count * np.dtype(dtype).itemsize),
            }
            self.context.set_tensor_address(name, int(entry["dev"]))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)

        _, _c, h, w = self.inputs[0]["shape"]
        inferred = "segment" if len(self.outputs) >= 2 else "detect"
        is_e2e = infer_e2e([o["shape"] for o in self.outputs], override=e2e)
        self.spec = ModelSpec(input_w=int(w), input_h=int(h), task=task or inferred, e2e=is_e2e)

    def run(self, input_tensor: np.ndarray):
        cuda = self._cuda
        inp = self.inputs[0]
        np.copyto(inp["host"], input_tensor.astype(inp["dtype"], copy=False).ravel())

        ev_start, ev_h2d, ev_exec, ev_d2h = (cuda.Event() for _ in range(4))
        ev_start.record(self.stream)
        cuda.memcpy_htod_async(inp["dev"], inp["host"], self.stream)
        ev_h2d.record(self.stream)
        self.context.execute_async_v3(self.stream.handle)
        ev_exec.record(self.stream)
        for o in self.outputs:
            cuda.memcpy_dtoh_async(o["host"], o["dev"], self.stream)
        ev_d2h.record(self.stream)
        self.stream.synchronize()

        outs = [o["host"].reshape(o["shape"]).astype(np.float32) for o in self.outputs]
        timing = DeviceTiming(
            dma_input_ms=ev_start.time_till(ev_h2d),
            compute_ms=ev_h2d.time_till(ev_exec),
            dma_output_ms=ev_exec.time_till(ev_d2h),
        )
        return outs, timing
