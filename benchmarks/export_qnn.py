"""Export Ultralytics YOLO to a Qualcomm QNN HTP context binary, calibrated on COCO train2017.

Host-side, in any environment with ``ultralytics`` and ``onnxruntime-qnn`` (x86-64
or aarch64 Linux, Python >= 3.11). Runs the stock Ultralytics ``format="qnn"``
export: ONNX Runtime QNN QDQ quantization to W8A16 (QUInt16 activations, QUInt8
weights, MinMax calibration), then offline compilation of the quantized graph into
a QNN context binary embedded in ``<model>_qnn.onnx`` for one Hexagon HTP target.
Calibration uses the same N seeded train2017 images as the other vendor lanes
(``export_tflite_int8.build_calibration_yaml``).

The artifact keeps Ultralytics' I/O contract — float32 NHWC input, boundary
Quantize/Dequantize on the CPU EP, classic one-to-many head (the QNN exporter
disables the yolo26 end-to-end head unless ``nms=False``) — and is run on-device
by ``qnn_infer.py``. The ``_qnn.onnx`` suffix is what routes it to Ultralytics'
QNN backend, so it is preserved.

Ultralytics selects the target with ``name`` (an HTP arch such as ``"73"`` or a
supported SoC such as ``"iq-8275"``). ``--soc-model`` re-points that name at a
specific QNN SoC model, for parts Ultralytics does not list — e.g. the Dragonwing
IQ-9075 (QCS9075, HTP v73, SoC model 77)::

    python -m benchmarks.export_qnn --name 73 --soc-model 77 \
        --models yolov8n yolov8n-seg --train-dir ~/coco/train2017 \
        --out-dir benchmarks/results/_models_qnn_iq9075
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from benchmarks.export_tflite_int8 import build_calibration_yaml

_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _safe_model(name: str) -> str:
    """Reject model names that could escape ``out_dir`` once used in a filename."""
    if not _MODEL_RE.fullmatch(name) or ".." in name:
        raise SystemExit(f"invalid model name {name!r}: expected [A-Za-z0-9._-]")
    return name


def set_target(name: str, soc_model: str | None) -> tuple[str, str]:
    """Resolve the Ultralytics QNN target, optionally overriding its SoC model."""
    from ultralytics.utils import QNN_HTP_TARGETS

    if soc_model is not None:
        QNN_HTP_TARGETS[name] = ("soc_model", str(soc_model))
    if name not in QNN_HTP_TARGETS:
        raise SystemExit(f"unknown QNN target {name!r}; known: {', '.join(QNN_HTP_TARGETS)}")
    return QNN_HTP_TARGETS[name]


def export_one(model: str, calib_yaml: Path, out_dir: Path, name: str) -> Path:
    """Export one model to a QNN context binary; return ``<out_dir>/<model>_qnn.onnx``."""
    import onnx
    from ultralytics import YOLO

    model = _safe_model(model)
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{model}_qnn.onnx"
    if dest.exists():
        print(f"[export] {dest} exists, skipping")
        return dest

    pt = model if model.endswith(".pt") else f"{model}.pt"
    print(f"[export] {pt} → QNN HTP (target {name}, calib {calib_yaml.name}) ...")
    # fraction=1.0 uses every image in the calib split; quantize is forced to w8a16.
    out = Path(YOLO(pt).export(format="qnn", name=name, data=str(calib_yaml),
                               fraction=1.0, imgsz=640, verbose=False))

    graph = onnx.load(str(out), load_external_data=False).graph
    ops = [n.op_type for n in graph.node]
    if ops.count("EPContext") != 1:
        raise RuntimeError(f"{out}: expected one EPContext node, got {ops}")
    shutil.move(str(out), dest)
    print(f"[export] → {dest} (ops: {sorted(set(ops))})")
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["yolov8n", "yolov8n-seg"])
    p.add_argument("--name", default="73", help="Ultralytics QNN target (HTP arch or SoC)")
    p.add_argument("--soc-model", default=None,
                   help="QNN SoC model to compile for, overriding the target's default")
    p.add_argument("--calib-n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-dir", default="/home/sebastien/coco/train2017")
    p.add_argument("--out-dir", default="benchmarks/results/_models_qnn")
    a = p.parse_args()

    option, value = set_target(a.name, a.soc_model)
    print(f"[export] QNN target {a.name}: {option}={value}")
    out_dir = Path(a.out_dir)
    calib_yaml = build_calibration_yaml(Path(a.train_dir), a.calib_n, a.seed, out_dir)
    for model in a.models:
        export_one(model, calib_yaml, out_dir, a.name)


if __name__ == "__main__":
    main()
