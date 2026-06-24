"""Export Ultralytics YOLO to full-integer INT8 TFLite, calibrated on COCO train2017.

Host-side, run inside the isolated ``venv-tfexport`` (TensorFlow + onnx2tf stack).
Produces a ``<model>-int8.tflite`` per model, calibrated on N images sampled from
the COCO **train2017** split (disjoint from the val2017 evaluation set). Optionally
runs the eIQ Neutron converter to produce an i.MX 95 ``-int8-imx95.tflite``.

The INT8 TFLite (UINT8 NHWC input, decoded box + class outputs, NMS-free) is the
artifact run on-device by ``tflite_infer.py``:
  - i.MX 8M Plus: the ``-int8.tflite`` directly, via the VX delegate.
  - i.MX 95: the Neutron-converted ``-int8-imx95.tflite``, via the Neutron delegate.

Usage (in venv-tfexport)::

    python -m benchmarks.export_tflite_int8 --models yolov8n yolov8n-seg \
        --calib-n 500 --train-dir /home/sebastien/coco/train2017 \
        --out-dir benchmarks/results/_models_tflite \
        --neutron ~/eiq-neutron-sdk-linux-3.0.1/bin/neutron-converter --neutron-target imx95
"""
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
from pathlib import Path

from benchmarks.coco_dataset import _write_dataset_yaml  # reuse the 80-class yaml writer


def build_calibration_yaml(train_dir: Path, n: int, seed: int, out_root: Path) -> Path:
    """Symlink N seeded-random train2017 images into a calib dir + write a dataset yaml.

    PTQ calibration only feeds pixels through the model, so labels are not needed
    (Ultralytics treats the missing labels dir as empty/background). Deterministic
    via the fixed seed so the quantization is reproducible.
    """
    train_dir = Path(train_dir).expanduser().resolve()
    out_root = Path(out_root).expanduser().resolve()
    calib_images = out_root / "calib" / "images" / "val2017"
    calib_images.mkdir(parents=True, exist_ok=True)

    jpgs = sorted(train_dir.glob("*.jpg"))
    if len(jpgs) < n:
        raise SystemExit(f"only {len(jpgs)} images in {train_dir}, need {n}")
    rng = random.Random(seed)
    picked = rng.sample(jpgs, n)
    for jpg in picked:
        link = calib_images / jpg.name
        if not link.exists():
            link.symlink_to(jpg)

    # val2017.txt list + yaml. val/train both point at the calib images so the
    # int8 calibration loader (split default "val", fraction 1.0) uses all N.
    (out_root / "calib" / "images" / "val2017.txt").write_text(
        "\n".join(str(p) for p in sorted(calib_images.glob("*.jpg"))) + "\n", encoding="utf-8")
    yaml_path = out_root / "calib" / "coco-calib.yaml"
    _write_dataset_yaml(yaml_path, out_root / "calib", "images/val2017.txt", "images/val2017.txt")
    print(f"[calib] {n} train2017 images (seed={seed}) → {yaml_path}")
    return yaml_path


def export_one(model: str, calib_yaml: Path, out_dir: Path) -> Path:
    """Export one model to INT8 TFLite; return the path to <model>-int8.tflite."""
    from ultralytics import YOLO

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{model}-int8.tflite"
    if dest.exists():
        print(f"[export] {dest} exists, skipping")
        return dest

    pt = model if model.endswith(".pt") else f"{model}.pt"
    print(f"[export] {pt} → INT8 TFLite (calib {calib_yaml.name}) ...")
    yolo = YOLO(pt)
    # int8=True → full-integer PTQ; data yaml supplies the representative dataset;
    # fraction=1.0 uses every image in the calib split.
    out = yolo.export(format="tflite", int8=True, data=str(calib_yaml),
                      fraction=1.0, imgsz=640, verbose=False)
    # onnx2tf emits several variants; the *full-integer* one has int8 IO (the
    # NPU delegates / Neutron need int8 activations end to end). Ultralytics'
    # `<stem>_int8.tflite` is a float-IO sibling, so pick `_full_integer_quant`.
    stem = Path(pt).stem
    produced = Path(out)
    roots = [produced if produced.is_dir() else produced.parent,
             Path(f"{stem}_saved_model"), Path(".")]
    cand = None
    for root in roots:
        hits = list(root.rglob(f"{stem}_full_integer_quant.tflite"))
        if hits:
            cand = hits[0]
            break
    if cand is None:
        raise RuntimeError(f"no {stem}_full_integer_quant.tflite found after export of {pt}")
    shutil.copy2(cand, dest)
    print(f"[export] → {dest}")
    return dest


def neutron_convert(int8_tflite: Path, converter: Path, target: str, out_dir: Path) -> Path:
    """Run the eIQ Neutron converter → <model>-int8-<target>.tflite."""
    dest = out_dir / f"{int8_tflite.stem}-{target}.tflite"
    if dest.exists():
        print(f"[neutron] {dest} exists, skipping")
        return dest
    cmd = [str(converter), "--input", str(int8_tflite), "--target", target, "--output", str(dest)]
    print(f"[neutron] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[neutron] → {dest}")
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["yolov8n", "yolov8n-seg"])
    p.add_argument("--calib-n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-dir", default="/home/sebastien/coco/train2017")
    p.add_argument("--out-dir", default="benchmarks/results/_models_tflite")
    p.add_argument("--neutron", default=None, help="path to neutron-converter binary")
    p.add_argument("--neutron-target", default="imx95")
    a = p.parse_args()

    out_dir = Path(a.out_dir)
    calib_yaml = build_calibration_yaml(Path(a.train_dir), a.calib_n, a.seed, out_dir)
    for model in a.models:
        int8 = export_one(model, calib_yaml, out_dir)
        if a.neutron:
            neutron_convert(int8, Path(a.neutron).expanduser(), a.neutron_target, out_dir)


if __name__ == "__main__":
    main()
