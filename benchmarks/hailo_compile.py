"""Compile an Ultralytics-exported detection model to a quantized INT8 Hailo .hef.

This is the **compile** half of the Ultralytics/Vendor Hailo workflow
(https://docs.ultralytics.com/integrations/hailo). On an x86_64 host with the
Hailo Dataflow Compiler (the ``venv-hailo`` built by ``hailo_install_dfc.sh``)
it turns ``yolo*.pt`` into a quantized INT8 ``yolo*.hef`` for the
``rpi5-hailo8l`` platform. On-device HailoRT inference + COCO scoring is a
separate step on the Pi. Detection only; NMS is baked into the ``.hef``.

GAP-FILL ROLE: the rpi5-hailo8l baseline primarily uses the Hailo Model Zoo's
PRECOMPILED HEFs (``hailo_fetch_zoo.sh`` — the vendor's published artifacts).
This script is for models the zoo does not publish (e.g. ``yolov5nu``, the
Ultralytics anchor-free v5 retrain), recorded as ``vendor:hailo-dfc``.

Pipeline per model (mirrors the Ultralytics doc, generated programmatically so
it generalizes across the anchor-free nano set yolov8n/yolo11n/yolov5nu)::

    .pt --(./venv ultralytics, subprocess)--> .onnx
        --translate_onnx_model(6 head end-nodes)--> HN graph
        --.alls (normalize + sigmoid on cls convs + nms_postprocess)
        --optimize(calibration: N train2017 images)--> INT8
        --compile()--> .hef

Why the ONNX export is delegated to ``./venv``: the DFC pins
numpy/onnx/protobuf and pulls TensorFlow, so it lives in an isolated venv that
deliberately has no torch/ultralytics. The benchmark ``./venv`` owns the export.

Run with ``venv-hailo/bin/python -m benchmarks.hailo_compile`` from the repo
root. Artifacts land in ``benchmarks/results/hailo/`` (gitignored).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

# The 6 detection-head end-nodes are the last conv of each cv2.* (box, 4*reg_max
# channels) and cv3.* (class, nc channels) branch, under the Detect layer
# /model.<N>/. This pattern holds for the anchor-free DFL head shared by
# yolov8 / yolo11 / yolov5u.
_HEAD_CONV = re.compile(r"^/model\.\d+/cv[23]\.\d+/cv[23]\.\d+\.2/Conv$")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "benchmarks" / "results" / "hailo"
DEFAULT_VENV_PY = REPO_ROOT / "venv" / "bin" / "python"
DEFAULT_COCO_TRAIN = Path.home() / "coco" / "train2017"


def export_onnx(model: str, out_dir: Path, venv_python: Path,
                imgsz: int = 640, opset: int = 11) -> Path:
    """Export ``model`` to ONNX via the benchmark ./venv (ultralytics+torch).

    Idempotent: skips export if the ONNX already exists.
    """
    onnx_path = out_dir / f"{model}.onnx"
    if onnx_path.exists():
        print(f"[export] {onnx_path.name} exists — reuse")
        return onnx_path
    if not venv_python.exists():
        raise SystemExit(f"benchmark venv python not found: {venv_python}")
    code = (
        "import os, shutil; os.environ['YOLO_AUTOINSTALL']='False';"
        "from pathlib import Path; from ultralytics import YOLO;"
        f"p=YOLO('{model}.pt').export(format='onnx', imgsz={imgsz}, opset={opset});"
        f"d=Path(r'{onnx_path}'); s=Path(p);"
        "shutil.copyfile(s, d) if s.resolve()!=d.resolve() else None;"
        "print('EXPORTED', d)"
    )
    print(f"[export] {model} -> {onnx_path.name} (via {venv_python})")
    subprocess.run([str(venv_python), "-c", code], check=True, cwd=out_dir)
    if not onnx_path.exists():
        raise SystemExit(f"export did not produce {onnx_path}")
    return onnx_path


def find_head(onnx_path: Path) -> tuple[str, list[str]]:
    """Return (input_tensor_name, [6 head end-node names]) from the ONNX."""
    import onnx

    m = onnx.load(str(onnx_path))
    input_name = m.graph.input[0].name
    ends = sorted(n.name for n in m.graph.node if _HEAD_CONV.match(n.name))
    if len(ends) != 6:
        raise SystemExit(
            f"expected 6 detection-head end-nodes, found {len(ends)} in "
            f"{onnx_path.name}: {ends}. Is this an anchor-free detect model?")
    return input_name, ends


def _decoders_from_hn(runner, imgsz: int):
    """Introspect the parsed HN: map each output to (stride, role, conv name).

    Returns (nc, reg_max, decoders, cls_convs) where ``decoders`` is a
    stride-keyed dict ``{stride: {"reg_layer": name, "cls_layer": name}}`` and
    ``cls_convs`` is the list of class-branch convs to sigmoid.
    """
    layers = runner.get_hn()["layers"]
    short = lambda full: full.split("/", 1)[1] if "/" in full else full  # noqa: E731
    per_stride: dict[int, dict] = {}
    nc = reg_max = None
    for name, spec in layers.items():
        if spec.get("type") != "output_layer":
            continue
        conv = spec["input"][0]  # the single conv feeding this output
        _, gh, _, ch = spec["output_shapes"][0]  # [-1, H, W, C], NHWC
        stride = imgsz // gh
        d = per_stride.setdefault(stride, {})
        if ch % 4 == 0 and ch >= 64 and ch in (64, 4 * 16):
            # regression (box) branch: channels = 4 * reg_max
            reg_max = ch // 4
            d["reg_layer"] = short(conv)
        else:
            # classification branch: channels = nc
            nc = ch
            d["cls_layer"] = short(conv)
    decoders = {s: per_stride[s] for s in sorted(per_stride)}
    cls_convs = [d["cls_layer"] for _, d in sorted(decoders.items())]
    return nc, reg_max, decoders, cls_convs


def build_scripts(runner, model: str, out_dir: Path, imgsz: int,
                  score_th: float, iou_th: float, max_per_class: int) -> str:
    """Write the per-model NMS config JSON and return the .alls model script."""
    nc, reg_max, decoders, cls_convs = _decoders_from_hn(runner, imgsz)
    nms_cfg = {
        "nms_scores_th": score_th,
        "nms_iou_th": iou_th,
        "image_dims": [imgsz, imgsz],
        "max_proposals_per_class": max_per_class,
        "classes": nc,
        "regression_length": reg_max,
        "background_removal": False,
        "bbox_decoders": [
            {"name": f"bbox_decoder_{s}", "stride": s,
             "reg_layer": d["reg_layer"], "cls_layer": d["cls_layer"]}
            for s, d in decoders.items()
        ],
    }
    cfg_path = out_dir / f"{model}_nms_config.json"
    cfg_path.write_text(json.dumps(nms_cfg, indent=2))
    print(f"[alls] {model}: nc={nc} reg_max={reg_max} strides={list(decoders)} "
          f"-> {cfg_path.name}")
    # Inputs are 0-255 RGB; normalization folds the /255 into the net. Class
    # branches get sigmoid (the Detect head's sigmoid was cut with the end-nodes).
    sig = "\n".join(f"change_output_activation({c}, sigmoid)" for c in cls_convs)
    alls = (
        "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"
        f"{sig}\n"
        f'nms_postprocess("{cfg_path}", meta_arch=yolov8, engine=cpu)\n'
    )
    (out_dir / f"{model}.alls").write_text(alls)
    return alls


def _letterbox(img, size: int) -> np.ndarray:
    """Resize a PIL RGB image keeping aspect ratio, pad to size×size (gray 114).

    Returns an (size, size, 3) float32 0-255 array — matching the on-device
    preprocessing so calibration sees the same activation distribution.
    """
    from PIL import Image

    w, h = img.size
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return np.asarray(canvas, dtype=np.float32)


def calibration_set(coco_train: Path, n: int, imgsz: int) -> np.ndarray:
    """Build an (n, imgsz, imgsz, 3) float32 0-255 RGB calibration tensor.

    Deterministic sample (sorted, evenly strided) of train2017 — disjoint from
    the val2017 eval set.
    """
    from PIL import Image

    files = sorted(coco_train.glob("*.jpg"))
    if not files:
        raise SystemExit(f"no calibration images under {coco_train}")
    if len(files) > n:
        step = len(files) / n
        files = [files[int(i * step)] for i in range(n)]
    print(f"[calib] {len(files)} images from {coco_train} @ {imgsz}")
    arr = np.zeros((len(files), imgsz, imgsz, 3), dtype=np.float32)
    for i, f in enumerate(files):
        with Image.open(f) as im:
            arr[i] = _letterbox(im.convert("RGB"), imgsz)
    return arr


def compile_model(model: str, out_dir: Path, venv_python: Path,
                  coco_train: Path, hw_arch: str, imgsz: int, opset: int,
                  calib_images: int, score_th: float, iou_th: float,
                  max_per_class: int) -> dict:
    from hailo_sdk_client import ClientRunner

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_onnx(model, out_dir, venv_python, imgsz, opset)
    input_name, ends = find_head(onnx_path)
    print(f"[translate] {model}: input={input_name} end_nodes={len(ends)}")

    runner = ClientRunner(hw_arch=hw_arch)
    runner.translate_onnx_model(
        str(onnx_path), model, end_node_names=ends,
        net_input_shapes={input_name: [1, 3, imgsz, imgsz]})

    alls = build_scripts(runner, model, out_dir, imgsz,
                         score_th, iou_th, max_per_class)
    runner.load_model_script(alls)

    calib = calibration_set(coco_train, calib_images, imgsz)
    # INT8 quantization. The DFC raises its optimization level (statistics →
    # equalization → QAT fine-tuning) with ≥1024 calibration images AND a GPU;
    # on a CPU-only host fine-tuning is skipped, so INT8 accuracy is somewhat
    # below a GPU-quantized build. See hailo_install_dfc.sh for the env.
    print(f"[optimize] {model}: INT8 quantization (this is the slow step)…")
    runner.optimize(calib)
    har_path = out_dir / f"{model}.har"
    runner.save_har(str(har_path))

    print(f"[compile] {model}: allocating for {hw_arch}…")
    hef = runner.compile()
    hef_path = out_dir / f"{model}.hef"
    hef_path.write_bytes(hef)
    size_mb = hef_path.stat().st_size / 1e6
    print(f"[done] {model}: {hef_path.name} ({size_mb:.2f} MB)")
    return {"model": model, "hef": hef_path.name, "har": har_path.name,
            "hw_arch": hw_arch, "imgsz": imgsz, "calib": f"train2017/{calib_images}",
            "size_mb": round(size_mb, 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["yolo11n"],
                    help="Ultralytics detection models (default: yolo11n)")
    ap.add_argument("--hw-arch", default="hailo8l",
                    choices=["hailo8", "hailo8l", "hailo15h"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=11)
    ap.add_argument("--calib-images", type=int, default=1024,
                    help="train2017 images for INT8 calibration (default 1024, "
                         "the DFC's recommended minimum; fewer drops the "
                         "optimization level and lowers INT8 accuracy)")
    ap.add_argument("--coco-train", type=Path, default=DEFAULT_COCO_TRAIN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--venv-python", type=Path, default=DEFAULT_VENV_PY,
                    help="benchmark venv python used for ONNX export")
    ap.add_argument("--score-th", type=float, default=0.001)
    ap.add_argument("--iou-th", type=float, default=0.7)
    ap.add_argument("--max-per-class", type=int, default=100)
    a = ap.parse_args()

    manifest = []
    for model in a.models:
        info = compile_model(
            model, a.out, a.venv_python, a.coco_train, a.hw_arch, a.imgsz,
            a.opset, a.calib_images, a.score_th, a.iou_th, a.max_per_class)
        manifest.append(info)
    man_path = a.out / "compile_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nCompiled {len(manifest)} model(s); manifest -> {man_path}")


if __name__ == "__main__":
    main()
