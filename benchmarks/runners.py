# benchmarks/runners.py
"""Benchmark runners for Ultralytics and yolo-validator.

Two entry points:
  run_ultralytics  — runs YOLO.val() with save_json=True, collects speed dict.
  run_yolo_validator — runs ValidationPipeline via on_frame callback for per-frame
                       timings and COCO prediction assembly.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

from yolo_validator.backends import load_backend
from yolo_validator.coco_output import coco80_to_coco91, detections_to_coco
from yolo_validator.pipeline import ValidationPipeline


def run_ultralytics(
    model_path: str,
    data_yaml: str,
    task: str,
    imgsz: int = 640,
    pre_val_model=None,
    device: str | int = "cpu",
) -> dict:
    """Run Ultralytics val and collect predictions + speed.

    Args:
        model_path: path to .pt or .onnx model.
        data_yaml: path to ultralytics dataset YAML (coco-val.yaml or subset_N.yaml).
        task: "detect" or "segment".
        imgsz: input image size (default 640).
        pre_val_model: optional pre-configured YOLO model object to use instead of
                       loading from model_path (e.g. with end2end already set).

    Returns:
        dict with keys:
          pred_json_path (str): path to predictions.json saved by ultralytics.
          predictions (list[dict]): loaded COCO-format predictions.
          speed (dict): {preprocess, inference, postprocess} mean ms/image.
          n_images (int): number of images evaluated.
          wall_s (float): wall-clock seconds for the entire val call.
    """
    from ultralytics import YOLO

    t0 = time.perf_counter()
    model = pre_val_model if pre_val_model is not None else YOLO(model_path)

    # rect=False forces square 640×640 letterbox, matching yolo-validator behaviour.
    # Ultralytics default rect=True uses rectangular inference which gives different
    # letterbox padding and would produce different predictions than yolo-validator.
    #
    # device selects execution target: "cpu" (default) or a CUDA index (e.g. 0)
    # for the Jetson/TensorRT runs. On CPU this matches yolo-validator's ONNX
    # Runtime CPU provider; for .engine models device must be a CUDA index.
    #
    # project is set to a temp dir to avoid cluttering other projects' runs/
    # directories (Ultralytics walks up looking for a project directory).
    with tempfile.TemporaryDirectory(prefix="ult_val_") as tmp_project:
        results = model.val(
            data=data_yaml,
            save_json=True,
            conf=0.001,
            iou=0.7,
            max_det=300,
            imgsz=imgsz,
            rect=False,
            batch=1,
            plots=False,
            verbose=False,
            device=device,
            project=tmp_project,
        )

        wall_s = time.perf_counter() - t0
        speed = dict(results.speed)  # {preprocess, inference, postprocess} mean ms/image

        # Ultralytics saves predictions.json in results.save_dir when save_json=True
        pred_json_path = Path(results.save_dir) / "predictions.json"
        if not pred_json_path.exists():
            raise FileNotFoundError(
                f"Expected Ultralytics predictions.json at {pred_json_path}. "
                "Check that save_json=True was accepted by this ultralytics version."
            )

        # Load predictions while temp dir still exists
        with open(pred_json_path, encoding="utf-8") as f:
            preds = json.load(f)

    # Persist predictions to a stable on-disk path (gitignored results/ tree, NOT
    # /tmp — these are expensive long-run outputs that must survive a reboot).
    key = hashlib.md5((data_yaml + str(model_path)).encode()).hexdigest()[:8]
    stable_dir = Path(__file__).resolve().parent / "results" / "_ult_preds" / key
    stable_dir.mkdir(parents=True, exist_ok=True)
    stable_pred_json = stable_dir / "predictions.json"
    with open(stable_pred_json, "w", encoding="utf-8") as f:
        json.dump(preds, f)

    n_images = len(set(p["image_id"] for p in preds)) if preds else 0

    return {
        "pred_json_path": str(stable_pred_json),
        "predictions": preds,
        "speed": speed,
        "n_images": n_images,
        "wall_s": wall_s,
    }


def run_yolo_validator(
    model_onnx: str,
    images_dir: str,
    path: str,
    task: str,
    gt_json: str,
    max_images: int | None = None,
    warmup: int = 3,
    runtime: str = "onnx",
    provider: str = "cpu",
) -> dict:
    """Run yolo_validator ValidationPipeline via on_frame callback.

    Args:
        model_onnx: path to ONNX model.
        images_dir: directory of *.jpg images to process.
        path: preprocess/postprocess path — "torch" or "numpy".
        task: "detect" or "segment" (used for mask handling).
        gt_json: path to COCO GT JSON (to look up image_id by filename).
        max_images: if set, limit to first N images (sorted).
        warmup: number of warmup frames (excluded from measurement).
        provider: ONNX execution provider — "cpu" or "cuda" (ignored for tensorrt runtime).

    Returns:
        dict with keys:
          predictions (list[dict]): COCO-format prediction dicts.
          stage_frame_timings (list[dict]): raw per-frame timing dicts for rebin.
          run_stats: RunStats object.
          n_images (int): number of measured images.
          wall_s (float): wall-clock seconds.
    """
    images_dir = Path(images_dir)
    gt_json = Path(gt_json)

    # Build filename → image_id mapping from GT JSON
    with open(gt_json, encoding="utf-8") as f:
        gt_data = json.load(f)
    filename_to_id = {Path(img["file_name"]).name: img["id"] for img in gt_data["images"]}

    # Collect all .jpg images, sorted for reproducibility
    image_paths = sorted(images_dir.glob("*.jpg"))
    if max_images is not None:
        image_paths = image_paths[:max_images]
    image_paths = [str(p) for p in image_paths]

    if not image_paths:
        raise ValueError(f"No .jpg images found in {images_dir}")

    # Build backend and pipeline; E2E models are auto-detected via backend.spec.e2e.
    # runtime="onnx" loads an ONNX model; runtime="tensorrt" loads a .engine.
    # provider is forwarded to the ONNX backend ("cpu" or "cuda"); tensorrt ignores it.
    backend = load_backend(model_onnx, runtime=runtime,
                           **({} if runtime != "onnx" else {"provider": provider}))
    pipeline = ValidationPipeline(
        backend,
        preprocess_path=path,
        postprocess_path=path,
        conf=0.001,
        iou=0.7,
        max_det=300,
        with_masks=(task == "segment"),
    )

    class_map = coco80_to_coco91()

    # Accumulators. Predictions are collected for ALL frames (warmup + measured)
    # so canonical_eval sees the same image set as Ultralytics; per-frame timings
    # are collected for MEASURED frames ONLY (warmup must not skew the stats).
    predictions: list[dict] = []
    stage_frame_timings: list[dict] = []

    def _collect_predictions(frame_result):
        fname = Path(frame_result.image_path).name
        image_id = filename_to_id.get(fname)
        if image_id is None:
            # Image not in GT — skip prediction (no ground truth to score against)
            return
        masks = frame_result.masks if task == "segment" else None
        predictions.extend(
            detections_to_coco(image_id, frame_result.detections, class_map, masks)
        )

    def on_frame(frame_result):
        t0 = time.perf_counter()
        _collect_predictions(frame_result)
        frame_result.timings["encode"] = (time.perf_counter() - t0) * 1e3
        stage_frame_timings.append(dict(frame_result.timings))  # measured only

    # warmup: predictions only (no timing) — keeps the scored image set identical
    # to Ultralytics while excluding cold-start frames from the timing stats.
    run_stats = pipeline.run(image_paths, warmup=warmup, on_frame=on_frame,
                             warmup_on_frame=_collect_predictions)

    return {
        "predictions": predictions,
        "stage_frame_timings": stage_frame_timings,
        "run_stats": run_stats,
        "n_images": run_stats.n_images,
        "wall_s": run_stats.wall_s,   # measured-loop wall (warmup excluded)
    }
