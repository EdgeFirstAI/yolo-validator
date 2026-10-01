"""On-device Ultralytics validation of a QNN HTP context binary + canonical COCO scoring.

Qualcomm targets run the Ultralytics validator itself: the ``<model>_qnn.onnx``
from ``export_qnn.py`` loads through Ultralytics' QNN backend (ONNX Runtime with
the ``onnxruntime-qnn`` plugin EP on the Hexagon HTP), and ``YOLO.val`` runs the
stock single-stream pipeline (batch 1, square 640 letterbox, conf 0.001, iou 0.7,
max_det 300). Predictions are re-scored by ``canonical_eval`` (crowd-as-normal),
like every other lane.

Ultralytics' QNN backend registers the plugin EP unconditionally, and ONNX
Runtime refuses a second registration in one process — run one model per
process.

Beyond Ultralytics' per-image preprocess/inference/postprocess means, validator
callbacks time the phases the speed dict leaves out: image load (the inline
dataloader — read, decode, letterbox; Ultralytics runs ``workers=0`` on CPU-device
backends), per-image loop time, and the setup / loop / finalize split of the
whole ``val`` call. Emits ``benchmark_a_<label>_<ts>.json`` (config key
``ult-qnn``) for ``benchmarks.normalize``.

Usage (on-device)::

    python -m benchmarks.qnn_infer --model yolov8n_qnn.onnx --label yolov8n \\
        --device-label iq9075-htp --coco-val ~/coco/val2017 \\
        --gt ~/coco/annotations/instances_val2017.json
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import time
from dataclasses import asdict
from pathlib import Path

from benchmarks.canonical_eval import canonical_eval
from benchmarks.coco_dataset import setup_shadow
from benchmarks.rebin import rebin_ultralytics
from benchmarks.runners import run_ultralytics
from yolo_validator._stats import stage_stats

_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _safe_label(name: str) -> str:
    if not _LABEL_RE.fullmatch(name) or ".." in name:
        raise SystemExit(f"invalid model label {name!r}: expected [A-Za-z0-9._-]")
    return name


class ValPhaseTimer:
    """Wall-clock phases of one ``YOLO.val`` call, from validator callbacks.

    The gap between one batch's end and the next batch's start is the dataloader
    fetch (image load); a batch's own span is preprocess + inference + postprocess
    + metric update.
    """

    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self.load_ms: list[float] = []
        self.batch_ms: list[float] = []
        self.loop_start = self.loop_end = None
        self._prev = self._batch = None

    def _on_val_start(self, _validator):
        self.loop_start = self._prev = self._clock()

    def _on_val_batch_start(self, _validator):
        self._batch = self._clock()
        self.load_ms.append((self._batch - self._prev) * 1e3)

    def _on_val_batch_end(self, _validator):
        self._prev = self._clock()
        self.batch_ms.append((self._prev - self._batch) * 1e3)

    def _on_val_end(self, _validator):
        self.loop_end = self._clock()

    @property
    def callbacks(self) -> dict:
        return {
            "on_val_start": self._on_val_start,
            "on_val_batch_start": self._on_val_batch_start,
            "on_val_batch_end": self._on_val_batch_end,
            "on_val_end": self._on_val_end,
        }

    def summary(self, t_call: float, t_done: float) -> dict:
        """Phase breakdown; ``t_call``/``t_done`` bracket the whole val call."""
        return {
            "image_load_ms": asdict(stage_stats(self.load_ms)),
            "batch_ms": asdict(stage_stats(self.batch_ms)),
            "val_phases_s": {
                "setup": self.loop_start - t_call,
                "loop": self.loop_end - self.loop_start,
                "finalize": t_done - self.loop_end,
            },
        }


def run(model_path, label, coco_val, gt, shadow_root, out_dir, device_label, imgsz=640):
    label = _safe_label(label)
    model_path = Path(model_path).expanduser().resolve()
    gt = Path(gt).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if not model_path.name.endswith("_qnn.onnx"):
        raise SystemExit(f"{model_path.name}: Ultralytics routes QNN models by the "
                         "'_qnn.onnx' suffix")
    for p in (model_path, gt):
        if not p.exists():
            raise SystemExit(f"not found: {p}")
    task = "segment" if "-seg" in label else "detect"

    shadow = setup_shadow(shadow_root, coco_val, gt)
    timer = ValPhaseTimer()
    print(f"[qnn_infer] {label}: {model_path.name}, task={task}")
    t_call = time.perf_counter()
    ult = run_ultralytics(str(model_path), str(shadow / "coco-val.yaml"), task,
                          imgsz=imgsz, callbacks=timer.callbacks)
    t_done = time.perf_counter()

    iou_types = ("bbox", "segm") if task == "segment" else ("bbox",)
    print(f"[qnn_infer] scoring {len(ult['predictions'])} dets…")
    metrics = canonical_eval(str(gt), ult["predictions"], iou_types=iou_types)
    n, wall = ult["n_images"], ult["wall_s"]
    cfg = {
        "bbox": metrics["bbox"], "segm": metrics.get("segm"),
        "timing": rebin_ultralytics(ult["speed"], n),
        "wall_s": wall, "n_images": n, "fps_wall": n / wall if wall else None,
        **timer.summary(t_call, t_done),
    }
    doc = {"label": label, "task": task,
           "host": {"machine": platform.machine(), "node": platform.node(),
                    "system": platform.platform(), "device": device_label},
           "configs": {"ult-qnn": cfg}}
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"benchmark_a_{label}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(doc, indent=2))
    b = metrics["bbox"]
    mask = f" mask AP={metrics['segm']['AP']:.4f}" if metrics.get("segm") else ""
    print(f"[qnn_infer] {label}: box AP={b['AP']:.4f} AP50={b['AP50']:.4f}{mask} "
          f"| {cfg['fps_wall']:.2f} fps | load {cfg['image_load_ms']['mean_ms']:.1f} / "
          f"inf {cfg['timing']['inference']:.1f} ms -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path, help="<model>_qnn.onnx")
    ap.add_argument("--label", required=True, help="variant label, e.g. yolov8n")
    ap.add_argument("--device-label", required=True, help="metrics platform id, e.g. iq9075-htp")
    ap.add_argument("--coco-val", type=Path, default=Path.home() / "coco" / "val2017")
    ap.add_argument("--gt", type=Path,
                    default=Path.home() / "coco" / "annotations" / "instances_val2017.json")
    ap.add_argument("--shadow-root", type=Path, default=Path("benchmarks/results/_shadow_coco"))
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results/qnn"))
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    run(a.model, a.label, a.coco_val, a.gt, a.shadow_root, a.out, a.device_label, a.imgsz)


if __name__ == "__main__":
    main()
