# benchmarks/benchmark_a.py
"""Benchmark A orchestrator and CLI.

Runs 4 configs per model variant:
  ult-pt    — Ultralytics val with .pt model
  ult-onnx  — Ultralytics val with .onnx model
  yv-torch  — yolo-validator ValidationPipeline, torch postprocess
  yv-numpy  — yolo-validator ValidationPipeline, numpy postprocess

On device=coreml (macOS Apple Silicon) the lane is:
  ult-onnx   — Ultralytics val on the FP32 .onnx graph (CPU EP) — the FP32
               accuracy anchor (edge precision is FP16, but the FP32 reference
               is the same .onnx the CPU/CUDA lanes use; see STATUS.md)
  ult-coreml — Ultralytics val on a native FP16 .mlpackage (CPU_AND_NE → ANE)
               — the genuine native-CoreML reference
  yv-torch   — yolo-validator on the ONNX Runtime CoreML EP (FP16, ANE) — proxy
  yv-numpy   — yolo-validator on the ONNX Runtime CoreML EP (FP16, ANE) — proxy
ult-pt is skipped on coreml (the deployment artifact is the .mlpackage / CoreML
EP, FP16; the FP32 PyTorch lane is the onnx-cpu/onnx-cuda anchor's job, not this
lane's). The yv proxy reuses the FP32 .onnx — the CoreML EP casts to FP16 at
load, so no separate FP16 ONNX export is needed.

For yolo26 models, two variants are produced:
  {name}-classic   — classic anchor-grid export (Detect.end2end=False)
  {name}-nmsfree   — E2E NMS-free export       (Detect.end2end=True)

For yolov8 models (no end2end attribute), only classic mode is available.

Outputs per variant:
  output_dir/benchmark_a_{label}_{timestamp}.md
  output_dir/benchmark_a_{label}_{timestamp}.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from benchmarks.canonical_eval import canonical_eval
from benchmarks.coco_dataset import make_subset, setup_shadow
from benchmarks.rebin import rebin_samples, rebin_ultralytics
from benchmarks.runners import run_ultralytics, run_yolo_validator
from benchmarks.studio_fetch import edgefirst_canonical


def _host_meta() -> dict:
    """Capture the host + thread environment so timing numbers are
    interpretable across machines (CPU ms/FPS depend on core/thread count)."""
    import os
    import platform
    meta = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "python": platform.python_version(),
    }
    try:
        import torch
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
            meta["cuda_version"] = torch.version.cuda
    except Exception:
        pass
    return meta


def _has_end2end(pt_path: str) -> bool:
    """Return True if the model is natively NMS-free (has a trained one2one head).

    Every Ultralytics ``Detect`` head carries an ``end2end`` attribute (default
    ``False``), so ``hasattr`` is True even for classic-only models like yolov8.
    The real discriminator is the *value*: yolo26/yolov10 load with
    ``end2end=True`` (and a ``one2one`` head); yolov8 loads with ``end2end=False``
    and cannot be exported NMS-free. Only natively-end2end models get both the
    classic and nms-free variants.
    """
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect

    model = YOLO(str(pt_path))
    for m in model.model.modules():
        if isinstance(m, Detect):
            return bool(getattr(m, "end2end", False))
    return False


def export_to_onnx(pt_path: str, output_dir: str, export_mode: str = "classic",
                   dynamic: bool = False, half: bool = False) -> str:
    """Export .pt to .onnx with the given export_mode.

    export_mode:
      "classic"  — set Detect.end2end=False before export (anchor-grid output)
      "nmsfree"  — set Detect.end2end=True before export (E2E NMS-free output)
    ``dynamic=True`` exports a dynamic batch axis (required for batch_size>1).
    half:
      False — FP32 graph (default).
      True  — FP16 graph (weights + IO in float16). Requires a CUDA device for
              the Ultralytics half export; run with --device cuda. The OnnxRuntime
              backend feeds float16 input and upcasts outputs to float32.

    Idempotent: returns existing file if present.

    Returns:
        Absolute path to .onnx file.
    """
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect

    pt_path = Path(pt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always include mode suffix so classic and nmsfree exports never collide,
    # and so pre-existing ONNX files from other sessions don't shadow fresh exports.
    # A -dyn suffix keeps the dynamic-batch export separate from the static one;
    # a -fp16 suffix keeps FP32 and FP16 graphs side by side (like .engine).
    suffix = (f"-{export_mode}" + ("-dyn" if dynamic else "")
              + ("-fp16" if half else ""))
    onnx_path = output_dir / (pt_path.stem + suffix + ".onnx")
    if onnx_path.exists():
        print(f"[export] {onnx_path} already exists, skipping export")
        return str(onnx_path)

    print(f"[export] exporting {pt_path} ({export_mode}, "
          f"{'fp16' if half else 'fp32'}) → {onnx_path} ...")
    model = YOLO(str(pt_path))
    for m in model.model.modules():
        if isinstance(m, Detect):
            if export_mode == "classic":
                m.end2end = False
            else:
                m.end2end = True
            break
    export_kwargs = {"format": "onnx", "imgsz": 640, "dynamic": dynamic}
    if half:
        # Ultralytics requires a CUDA device to export a half-precision graph.
        export_kwargs.update(half=True, device=0)
    exported = model.export(**export_kwargs)
    exported_path = Path(exported)
    if exported_path != onnx_path:
        exported_path.rename(onnx_path)
    print(f"[export] done: {onnx_path}")
    return str(onnx_path)


def export_to_engine(pt_path: str, output_dir: str, export_mode: str = "classic",
                     half: bool = False) -> str:
    """Export .pt to a TensorRT .engine (device-specific; build on the target).

    Mode is applied via Detect.end2end before export (classic = anchor-grid +
    NMS; nmsfree = E2E head). The engine is built by Ultralytics (which goes
    .pt → .onnx → TRT). Both Ultralytics and yolo-validator load this same
    engine (yolo-validator strips the Ultralytics metadata header). Idempotent.

    Returns absolute path to the .engine file.
    """
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect

    pt_path = Path(pt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"-{export_mode}" + ("-fp16" if half else "")
    engine_path = output_dir / (pt_path.stem + suffix + ".engine")
    if engine_path.exists():
        print(f"[export] {engine_path} already exists, skipping export")
        return str(engine_path)

    print(f"[export] building TensorRT engine {pt_path} ({export_mode}, "
          f"{'fp16' if half else 'fp32'}) → {engine_path} ...")
    model = YOLO(str(pt_path))
    for m in model.model.modules():
        if isinstance(m, Detect):
            m.end2end = (export_mode == "nmsfree")
            break
    exported = model.export(format="engine", imgsz=640, half=half,
                            device=0, dynamic=False, batch=1, verbose=False)
    exported_path = Path(exported)
    if exported_path != engine_path:
        exported_path.rename(engine_path)
    print(f"[export] done: {engine_path}")
    return str(engine_path)


def export_to_coreml(pt_path: str, output_dir: str, export_mode: str = "classic",
                     half: bool = True) -> str:
    """Export .pt to a native CoreML .mlpackage (macOS Apple Silicon).

    This is the genuine native-CoreML reference artifact validated by Ultralytics
    on-device (``YOLO("model.mlpackage").val(...)`` runs via coremltools, default
    ComputeUnit CPU_AND_NE → ANE). ``half=True`` exports FP16, matching how the
    edge lane deploys (STATUS.md: edge precision is FP16 + INT, never FP32).

    NMS is NEVER embedded (``nms=False``), even for classic heads. Embedding the
    CoreML NMS pipeline bakes in a fixed conf/IoU and a capped detection count,
    which makes a COCO val at conf=0.001 impossible — recall is clipped so mAP
    collapses (measured: yolov8n 0.344 embedded vs 0.45 raw), and the seg val
    path errors on the NMS-pipeline output layout. A validation artifact must
    emit the RAW model outputs so the Ultralytics validator applies NMS in
    postprocess at the parity conf/iou. ``export_mode`` still selects the head:
    classic = Detect.end2end=False (raw anchor-grid, external NMS); nmsfree =
    end2end=True (one2one, already end-to-end).

    The mode suffix (and -fp16/-fp32) is always in the filename so classic and
    nmsfree exports never collide and pre-existing packages don't shadow a fresh
    export. Idempotent: returns the existing package if present.

    Returns absolute path to the .mlpackage directory.
    """
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect

    pt_path = Path(pt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"-{export_mode}" + ("-fp16" if half else "-fp32")
    coreml_path = output_dir / (pt_path.stem + suffix + ".mlpackage")
    if coreml_path.exists():
        print(f"[export] {coreml_path} already exists, skipping export")
        return str(coreml_path)

    # NEVER embed NMS for a validation artifact: an embedded CoreML NMS pipeline
    # caps detections and fixes conf/IoU, which guts val mAP at conf=0.001 (see
    # docstring). Raw outputs → the Ultralytics validator applies NMS in postprocess.
    embed_nms = False
    print(f"[export] exporting CoreML {pt_path} ({export_mode}, "
          f"{'fp16' if half else 'fp32'}, nms={embed_nms}) → {coreml_path} ...")
    model = YOLO(str(pt_path))
    for m in model.model.modules():
        if isinstance(m, Detect):
            m.end2end = (export_mode == "nmsfree")
            break
    # device="mps" runs the export's conversion pass on Apple Metal; the produced
    # .mlpackage is the same regardless and is validated on CPU_AND_NE (ANE).
    exported = model.export(format="coreml", imgsz=640, half=half,
                            nms=embed_nms, device="mps")
    exported_path = Path(exported)
    if exported_path != coreml_path:
        exported_path.rename(coreml_path)
    print(f"[export] done: {coreml_path}")
    return str(coreml_path)


def _infer_task(model_name: str) -> str:
    return "segment" if "seg" in model_name else "detect"


def _safe_na(value, fmt=".3f") -> str:
    # pycocotools returns -1.0 for a size bucket (small/medium/large) with no
    # ground truth; render it as N/A rather than a misleading "-1.000".
    if value is None or (isinstance(value, (int, float)) and value < 0):
        return "N/A"
    return format(value, fmt)


def _sep_for(header: str) -> str:
    """Build a markdown separator row that matches the header's pipe layout
    exactly (avoids phantom columns from a hand-maintained separator)."""
    parts = header.split("|")
    return "|".join(p if i in (0, len(parts) - 1) else "-" * max(3, len(p))
                    for i, p in enumerate(parts))


def _fps_cell(r: dict) -> str:
    """Return the FPS cell value for a config row.

    Prefers fps_wall (true end-to-end wall-clock throughput). Falls back to
    fps_pipeline for configs that have no wall clock (e.g. edgefirst-profiler, whose
    Studio job time includes cloud overhead and is not comparable).
    Fallback values are marked with † so readers know the metric differs.
    """
    if r.get("fps_wall") is not None:
        return _safe_na(r["fps_wall"], ".1f")
    if r.get("fps_pipeline") is not None:
        return _safe_na(r["fps_pipeline"], ".1f") + "†"
    return "N/A"


def _build_markdown_table(results_per_config: dict, label: str) -> str:
    """Build the comparison markdown table for one variant."""
    # ult-coreml is the native FP16 .mlpackage reference present only on the
    # coreml device; it sits next to ult-onnx (the FP32 anchor) when present.
    base_configs = ["ult-pt", "ult-onnx", "ult-coreml", "yv-torch", "yv-numpy"]
    # Append any edgefirst-profiler configs present in results (order-stable)
    edgefirst_configs = [c for c in results_per_config if c.startswith("edgefirst-")]
    configs = base_configs + edgefirst_configs

    has_segm = any(
        results_per_config.get(c, {}).get("segm") is not None
        for c in base_configs
    )

    header = (
        "| Config     | box mAP | box mAP50 |"
        + (" mask mAP | mask mAP50 |" if has_segm else "")
        + " preprocess ms | inference ms | postprocess ms | e2e ms | wall s | FPS |"
    )
    sep = _sep_for(header)

    lines = [f"## Benchmark A — {label}", "", header, sep]

    ult_onnx = results_per_config.get("ult-onnx", {})
    has_fps_pipeline_fallback = False

    for config in configs:
        r = results_per_config.get(config, {})
        if not r:
            continue
        if "error" in r:
            lines.append(f"| {config:<8} | ERROR: {r['error'][:60]} |")
            continue

        bbox = r.get("bbox", {})
        segm = r.get("segm")
        timing = r.get("timing", {})

        box_map = _safe_na(bbox.get("AP"))
        box_map50 = _safe_na(bbox.get("AP50"))
        # segm AP: show — for configs that lack COCO seg AP (e.g. edgefirst-profiler)
        mask_map = _safe_na(segm.get("AP") if segm else None)
        mask_map50 = _safe_na(segm.get("AP50") if segm else None)
        pre = _safe_na(timing.get("preprocess"), ".1f")
        inf = _safe_na(timing.get("inference"), ".1f")
        post = _safe_na(timing.get("postprocess"), ".1f")
        e2e = _safe_na(timing.get("e2e"), ".1f")
        # wall s: show — for configs without a comparable local wall clock
        wall = "—" if r.get("wall_s") is None else _safe_na(r["wall_s"], ".1f")
        fps = _fps_cell(r)
        if fps.endswith("†"):
            has_fps_pipeline_fallback = True

        row = f"| {config:<10} | {box_map:>7} | {box_map50:>9} |"
        if has_segm:
            row += f" {mask_map:>8} | {mask_map50:>10} |"
        row += f" {pre:>13} | {inf:>12} | {post:>14} | {e2e:>6} | {wall:>6} | {fps:>5} |"
        lines.append(row)

    if has_fps_pipeline_fallback:
        lines.append("")
        lines.append("> † FPS shown is `fps_pipeline` (realized pipelined throughput "
                     "from the on-target profiler, using parallelism across capture / "
                     "inference / postprocess stages). Per-frame latency is in the "
                     "e2e column. Studio job wall time includes cloud queue overhead "
                     "and is not shown.")

    # Delta rows vs ult-onnx (base configs only — edgefirst-profiler excluded)
    if ult_onnx and "error" not in ult_onnx:
        lines.append("")
        lines.append("### Δ vs ult-onnx")
        lines.append("")
        lines.append(header.replace("Config", "Config (Δ)"))
        lines.append(sep)

        def delta(a, b, fmt=".3f"):
            if a is None or b is None:
                return "N/A"
            return format(a - b, "+" + fmt)

        ref_bbox = ult_onnx.get("bbox", {})
        ref_segm = ult_onnx.get("segm")
        ref_timing = ult_onnx.get("timing", {})

        for config in ["yv-torch", "yv-numpy"]:
            r = results_per_config.get(config, {})
            if not r or "error" in r:
                continue
            bbox = r.get("bbox", {})
            segm = r.get("segm")
            timing = r.get("timing", {})

            box_map_d = delta(bbox.get("AP"), ref_bbox.get("AP"))
            box_map50_d = delta(bbox.get("AP50"), ref_bbox.get("AP50"))
            mask_map_d = delta(
                segm.get("AP") if segm else None,
                ref_segm.get("AP") if ref_segm else None,
            )
            mask_map50_d = delta(
                segm.get("AP50") if segm else None,
                ref_segm.get("AP50") if ref_segm else None,
            )
            pre_d = delta(timing.get("preprocess"), ref_timing.get("preprocess"), ".1f")
            inf_d = delta(timing.get("inference"), ref_timing.get("inference"), ".1f")
            post_d = delta(timing.get("postprocess"), ref_timing.get("postprocess"), ".1f")
            e2e_d = delta(timing.get("e2e"), ref_timing.get("e2e"), ".1f")
            wall_d = delta(r.get("wall_s"), ult_onnx.get("wall_s"), ".1f")

            row = f"| {config + ' Δ':<10} | {box_map_d:>7} | {box_map50_d:>9} |"
            if has_segm:
                row += f" {mask_map_d:>8} | {mask_map50_d:>10} |"
            row += f" {pre_d:>13} | {inf_d:>12} | {post_d:>14} | {e2e_d:>6} | {wall_d:>6} | {'—':>5} |"
            lines.append(row)

    return "\n".join(lines) + "\n"


def run_benchmark_a(
    label: str,
    pt_path: str,
    task: str,
    export_mode: str,
    shadow_root: str | Path,
    images_dir: str | Path,
    gt_json: str | Path,
    output_dir: str | Path,
    max_images: int | None = None,
    warmup: int = 3,
    device: str = "cpu",
    half: bool = False,
    skip_ult_pt: bool = False,
    edgefirst_session_id: str | None = None,
    batch: int = 1,
) -> dict:
    """Run the benchmark configs for one model variant.

    device="cpu" runs ult-pt, ult-onnx, yv-torch, yv-numpy (CPU). device="cuda"
    runs the same 4 configs on the GPU. device="tensorrt" runs ult-pt (CUDA gold
    ref), ult-engine and yv-tensorrt on the same TensorRT engine (Jetson / NVIDIA
    GPU); ``half`` selects FP16 engine build. device="coreml" (macOS Apple
    Silicon) runs ult-onnx (FP32 anchor, CPU EP), ult-coreml (native FP16
    .mlpackage on ANE) and yv-torch/yv-numpy on the ONNX Runtime CoreML EP (FP16,
    ANE); ult-pt is skipped on coreml.

    Args:
        label: display name, e.g. "yolov8n-seg" or "yolo26n-classic".
        pt_path: path to .pt model.
        task: "detect" or "segment".
        export_mode: "classic" or "nmsfree".
        shadow_root: shadow COCO dataset root.
        images_dir: COCO val2017 images directory.
        gt_json: path to instances_val2017.json.
        output_dir: directory for output files.
        max_images: limit to first N images; None = all 5000.
        warmup: warmup frames for yolo-validator (excluded from timing).

    Returns:
        dict with per-config results.
    """
    from ultralytics import YOLO as _YOLO
    from ultralytics.nn.modules import Detect

    shadow_root = Path(shadow_root)
    images_dir = Path(images_dir).expanduser()
    gt_json = Path(gt_json).expanduser()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    iou_types = ("bbox", "segm") if task == "segment" else ("bbox",)

    # Guard: warmup must be < max_images
    effective_warmup = warmup
    if max_images is not None and max_images <= warmup:
        effective_warmup = max(0, max_images - 1)
        print(f"  [warn] max_images={max_images} <= warmup={warmup}; reducing warmup to {effective_warmup}")

    # ---- Ensure shadow dataset exists ----
    shadow_root = setup_shadow(shadow_root, images_dir, gt_json)

    # ---- Determine subset or full dataset ----
    if max_images is not None:
        subset_images_dir, subset_gt_json, subset_yaml = make_subset(
            shadow_root, images_dir, gt_json, max_images
        )
        run_gt_json = str(subset_gt_json)
        run_images_dir = str(subset_images_dir)
        run_yaml = str(subset_yaml)
    else:
        run_gt_json = str(gt_json)
        run_images_dir = str(shadow_root / "images" / "val2017")
        run_yaml = str(shadow_root / "coco-val.yaml")

    # ---- Build model artifacts + run configs (device-dependent) ----
    models_dir = output_dir / "_models"
    results_per_config: dict[str, dict] = {}

    def _set_end2end(model_obj, mode: str):
        """Apply export_mode to a loaded YOLO model's Detect head."""
        for m in model_obj.model.modules():
            if isinstance(m, Detect):
                m.end2end = (mode == "nmsfree")
                break

    if device == "tensorrt":
        engine_path = export_to_engine(str(pt_path), str(models_dir), export_mode, half=half)

        # ult-pt (CUDA) — PyTorch FP32 gold reference, on-device. Skippable:
        # where the TensorRT engine is the deployment path (Orin), the PyTorch
        # lane is redundant (pt ≡ engine ≤ 0.0001, proven) and very slow.
        if not skip_ult_pt:
            print(f"\n[{label}] running ult-pt (cuda) ...")
            try:
                ult_model = _YOLO(str(pt_path))
                _set_end2end(ult_model, export_mode)
                ult_pt = run_ultralytics(str(pt_path), run_yaml, task,
                                         pre_val_model=ult_model, device=0)
                results_per_config["ult-pt"] = {
                    **canonical_eval(run_gt_json, ult_pt["predictions"], iou_types),
                    "timing": rebin_ultralytics(ult_pt["speed"], ult_pt["n_images"]),
                    "n_images": ult_pt["n_images"],
                    "wall_s": ult_pt["wall_s"],
                    "speed": ult_pt["speed"],
                }
                print(f"  box AP={results_per_config['ult-pt'].get('bbox', {}).get('AP', 'N/A'):.4f}  wall={ult_pt['wall_s']:.1f}s")
            except Exception as e:
                print(f"  [FAILED] ult-pt: {e}")
                results_per_config["ult-pt"] = {"error": str(e)}

        # ult-engine (TensorRT) — Ultralytics val on the .engine
        print(f"\n[{label}] running ult-engine (tensorrt) ...")
        try:
            ult_eng = run_ultralytics(engine_path, run_yaml, task, device=0)
            results_per_config["ult-engine"] = {
                **canonical_eval(run_gt_json, ult_eng["predictions"], iou_types),
                "timing": rebin_ultralytics(ult_eng["speed"], ult_eng["n_images"]),
                "n_images": ult_eng["n_images"],
                "wall_s": ult_eng["wall_s"],
                "speed": ult_eng["speed"],
            }
            print(f"  box AP={results_per_config['ult-engine'].get('bbox', {}).get('AP', 'N/A'):.4f}  wall={ult_eng['wall_s']:.1f}s")
        except Exception as e:
            print(f"  [FAILED] ult-engine: {e}")
            results_per_config["ult-engine"] = {"error": str(e)}

        # yv-tensorrt — yolo-validator TensorRT backend on the same engine (numpy pre/post)
        print(f"\n[{label}] running yv-tensorrt ...")
        try:
            yv_trt = run_yolo_validator(
                engine_path, run_images_dir, "numpy", task, run_gt_json, max_images,
                warmup=effective_warmup, runtime="tensorrt",
            )
            yv_trt_stats = rebin_samples(yv_trt["stage_frame_timings"], detail=True)
            results_per_config["yv-tensorrt"] = {
                **canonical_eval(run_gt_json, yv_trt["predictions"], iou_types),
                "timing": {k: v.mean_ms for k, v in yv_trt_stats.items()},
                "timing_stats": {k: vars(v) for k, v in yv_trt_stats.items()},
                "n_images": yv_trt["n_images"],
                "wall_s": yv_trt["wall_s"],
            }
            print(f"  box AP={results_per_config['yv-tensorrt'].get('bbox', {}).get('AP', 'N/A'):.4f}  wall={yv_trt['wall_s']:.1f}s")
        except Exception as e:
            print(f"  [FAILED] yv-tensorrt: {e}")
            results_per_config["yv-tensorrt"] = {"error": str(e)}

    else:
        # device="cpu" | "cuda" | "coreml" — the yolo-validator proxy lane is the
        # same FP32 .onnx through different ONNX Runtime execution providers; only
        # the hardware target and the Ultralytics reference differ.
        #   cpu    : Ultralytics device="cpu", ORT CPUExecutionProvider.
        #   cuda   : Ultralytics device=0 (GPU), ORT CUDAExecutionProvider.
        #            Pre/postprocess stay on the CPU so the timing delta isolates
        #            inference speed. --half exports a FP16 graph and runs both the
        #            Ultralytics and yolo-validator lanes in FP16 (the OnnxRuntime
        #            backend feeds float16 input and upcasts outputs). --half on cpu
        #            is rejected in main() since ORT CPU has no useful FP16 path.
        #   coreml : the macOS Apple Silicon lane (FP16 deployment). ult-onnx runs
        #            on the CPU EP as the FP32 accuracy ANCHOR (the same .onnx the
        #            cpu/cuda anchors use — edge precision is FP16, but the anchor
        #            is FP32 by design; STATUS.md). yv-torch/yv-numpy run on the
        #            ORT CoreML EP (provider="coreml"), which casts the FP32 graph
        #            to FP16 and dispatches to the ANE — the runtime-matched twin
        #            of EdgeFirst's macos-onnx-coreml-ane lane. A native FP16
        #            .mlpackage reference (ult-coreml) is added below. ult-pt is
        #            skipped: the deployment artifact here is CoreML/FP16, and the
        #            FP32 PyTorch lane is the onnx-cpu/cuda anchor's job.
        _is_coreml = device == "coreml"
        if device == "cuda":
            _ult_device, _yv_provider = 0, "cuda"
        elif _is_coreml:
            # ult-onnx (the FP32 anchor) runs on the CPU EP; the proxy uses CoreML.
            _ult_device, _yv_provider = "cpu", "coreml"
        else:
            _ult_device, _yv_provider = "cpu", "cpu"
        # FP16 is a CUDA-only ONNX lane: export a half graph and run both the
        # Ultralytics and yolo-validator lanes in FP16. coreml has its own native
        # FP16 path (.mlpackage + CoreML EP), and --half on cpu is rejected in
        # main(), so use_half gates strictly on cuda.
        use_half = half and device == "cuda"

        # ---- Export to ONNX (FP32 anchor + yv proxy lanes; FP16 only on cuda) ----
        # batch>1 needs a dynamic-batch ONNX (a static batch=1 graph rejects (N,...)).
        onnx_path = export_to_onnx(str(pt_path), str(models_dir), export_mode,
                                   dynamic=(batch > 1), half=use_half)

        # ---- Config 1: ult-pt (skippable via --skip-ult-pt; auto-skip on coreml) ----
        # On coreml the FP32 PyTorch lane is redundant with the onnx-cpu/cuda FP32
        # anchor and is not part of this FP16 lane, so it is always skipped.
        if not skip_ult_pt and not _is_coreml:
            print(f"\n[{label}] running ult-pt ...")
            try:
                ult_model = _YOLO(str(pt_path))
                _set_end2end(ult_model, export_mode)
                ult_pt = run_ultralytics(str(pt_path), run_yaml, task,
                                          pre_val_model=ult_model, device=_ult_device,
                                          batch=batch, half=use_half)
                ult_pt_metrics = canonical_eval(run_gt_json, ult_pt["predictions"], iou_types)
                ult_pt_timing = rebin_ultralytics(ult_pt["speed"], ult_pt["n_images"])
                results_per_config["ult-pt"] = {
                    **ult_pt_metrics,
                    "timing": ult_pt_timing,
                    "n_images": ult_pt["n_images"],
                    "wall_s": ult_pt["wall_s"],
                    "speed": ult_pt["speed"],
                }
                print(f"  box AP={ult_pt_metrics.get('bbox', {}).get('AP', 'N/A'):.4f}  wall={ult_pt['wall_s']:.1f}s")
            except Exception as e:
                print(f"  [FAILED] ult-pt: {e}")
                results_per_config["ult-pt"] = {"error": str(e)}

        # ---- Config 2: ult-onnx ----
        print(f"\n[{label}] running ult-onnx ...")
        try:
            ult_onnx = run_ultralytics(onnx_path, run_yaml, task, device=_ult_device,
                                       batch=batch, half=use_half)
            ult_onnx_metrics = canonical_eval(run_gt_json, ult_onnx["predictions"], iou_types)
            ult_onnx_timing = rebin_ultralytics(ult_onnx["speed"], ult_onnx["n_images"])
            results_per_config["ult-onnx"] = {
                **ult_onnx_metrics,
                "timing": ult_onnx_timing,
                "n_images": ult_onnx["n_images"],
                "wall_s": ult_onnx["wall_s"],
                "speed": ult_onnx["speed"],
            }
            print(f"  box AP={ult_onnx_metrics.get('bbox', {}).get('AP', 'N/A'):.4f}  wall={ult_onnx['wall_s']:.1f}s")
        except Exception as e:
            print(f"  [FAILED] ult-onnx: {e}")
            results_per_config["ult-onnx"] = {"error": str(e)}

        # ---- Config 2b: ult-coreml (coreml only) — native FP16 .mlpackage on ANE ----
        # The genuine native-CoreML reference: Ultralytics val on a .mlpackage,
        # which coremltools runs on CPU_AND_NE (→ ANE) by default. This is the
        # right counterpart to EdgeFirst's macos-onnx-coreml-ane for accuracy
        # (box/mask mAP). FP16 by design (half=True). NMS is embedded for classic
        # heads and native for nmsfree heads (see export_to_coreml).
        if _is_coreml:
            print(f"\n[{label}] running ult-coreml (native .mlpackage, ANE) ...")
            try:
                coreml_path = export_to_coreml(str(pt_path), str(models_dir),
                                               export_mode, half=True)
                ult_cml = run_ultralytics(coreml_path, run_yaml, task)
                ult_cml_metrics = canonical_eval(run_gt_json, ult_cml["predictions"], iou_types)
                ult_cml_timing = rebin_ultralytics(ult_cml["speed"], ult_cml["n_images"])
                results_per_config["ult-coreml"] = {
                    **ult_cml_metrics,
                    "timing": ult_cml_timing,
                    "n_images": ult_cml["n_images"],
                    "wall_s": ult_cml["wall_s"],
                    "speed": ult_cml["speed"],
                }
                print(f"  box AP={ult_cml_metrics.get('bbox', {}).get('AP', 'N/A'):.4f}  wall={ult_cml['wall_s']:.1f}s")
            except Exception as e:
                print(f"  [FAILED] ult-coreml: {e}")
                results_per_config["ult-coreml"] = {"error": str(e)}

        # ---- Config 3: yv-torch ----
        print(f"\n[{label}] running yv-torch ...")
        try:
            yv_torch = run_yolo_validator(
                onnx_path, run_images_dir, "torch", task, run_gt_json, max_images,
                warmup=effective_warmup, provider=_yv_provider,
                batch_size=batch,
            )
            yv_torch_metrics = canonical_eval(run_gt_json, yv_torch["predictions"], iou_types)
            yv_torch_stats = rebin_samples(yv_torch["stage_frame_timings"], detail=True)
            yv_torch_timing = {k: v.mean_ms for k, v in yv_torch_stats.items()}
            results_per_config["yv-torch"] = {
                **yv_torch_metrics,
                "timing": yv_torch_timing,
                "timing_stats": {k: vars(v) for k, v in yv_torch_stats.items()},
                "n_images": yv_torch["n_images"],
                "wall_s": yv_torch["wall_s"],
            }
            print(f"  box AP={yv_torch_metrics.get('bbox', {}).get('AP', 'N/A'):.4f}  wall={yv_torch['wall_s']:.1f}s")
        except Exception as e:
            print(f"  [FAILED] yv-torch: {e}")
            results_per_config["yv-torch"] = {"error": str(e)}

        # ---- Config 4: yv-numpy ----
        print(f"\n[{label}] running yv-numpy ...")
        try:
            yv_numpy = run_yolo_validator(
                onnx_path, run_images_dir, "numpy", task, run_gt_json, max_images,
                warmup=effective_warmup, provider=_yv_provider,
                batch_size=batch,
            )
            yv_numpy_metrics = canonical_eval(run_gt_json, yv_numpy["predictions"], iou_types)
            yv_numpy_stats = rebin_samples(yv_numpy["stage_frame_timings"], detail=True)
            yv_numpy_timing = {k: v.mean_ms for k, v in yv_numpy_stats.items()}
            results_per_config["yv-numpy"] = {
                **yv_numpy_metrics,
                "timing": yv_numpy_timing,
                "timing_stats": {k: vars(v) for k, v in yv_numpy_stats.items()},
                "n_images": yv_numpy["n_images"],
                "wall_s": yv_numpy["wall_s"],
            }
            print(f"  box AP={yv_numpy_metrics.get('bbox', {}).get('AP', 'N/A'):.4f}  wall={yv_numpy['wall_s']:.1f}s")
        except Exception as e:
            print(f"  [FAILED] yv-numpy: {e}")
            results_per_config["yv-numpy"] = {"error": str(e)}

    # ---- Config: edgefirst-profiler (optional — fetched from EdgeFirst Studio) ----
    if edgefirst_session_id:
        print(f"\n[{label}] fetching edgefirst-profiler metrics from Studio "
              f"({edgefirst_session_id}) ...")
        try:
            edgefirst_result = edgefirst_canonical(edgefirst_session_id)
            results_per_config["edgefirst-profiler"] = edgefirst_result
            bbox_ap = (edgefirst_result.get("bbox") or {}).get("AP")
            fps_pipe = edgefirst_result.get("fps_pipeline")
            print(f"  box AP={bbox_ap:.4f}  fps_pipeline={fps_pipe:.1f}"
                  if bbox_ap is not None and fps_pipe is not None
                  else f"  session={edgefirst_result.get('session_name')}")
        except Exception as e:
            print(f"  [FAILED] edgefirst-profiler: {e}")
            results_per_config["edgefirst-profiler"] = {"error": str(e)}

    # ---- Derive FPS in code (one definition each, stored in the JSON) ----
    # fps_wall  = images / measured wall-clock seconds (true end-to-end
    #             throughput, single stream, batch=1 — the headline number).
    # fps_pipeline = 1000 / mean model-path e2e ms (steady-state per-frame rate,
    #             excludes image load/decode; > fps_wall by the I/O + Python gap).
    for r in results_per_config.values():
        if "error" in r:
            continue
        r["batch"] = batch
        wall = r.get("wall_s")
        n = r.get("n_images")
        if wall and n:
            r["fps_wall"] = n / wall
        e2e_ms = (r.get("timing") or {}).get("e2e")
        if e2e_ms and "fps_pipeline" not in r:
            r["fps_pipeline"] = 1000.0 / e2e_ms

    # ---- Write outputs ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"benchmark_a_{label}_{timestamp}"

    md_content = _build_markdown_table(results_per_config, label)
    md_path = output_dir / (stem + ".md")
    # encoding="utf-8" is mandatory: the report contains an em-dash and box
    # characters, and headless/SSH targets (e.g. Jetson) often run under an
    # ASCII locale where the default open() encoding would raise on write.
    md_path.write_text(md_content, encoding="utf-8")
    print(f"\n[{label}] wrote {md_path}")

    json_path = output_dir / (stem + ".json")
    full_results = {
        "label": label,
        "task": task,
        "export_mode": export_mode,
        "max_images": max_images,
        "timestamp": timestamp,
        "host": _host_meta(),   # so ms/FPS numbers are interpretable/reproducible
        "configs": results_per_config,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"[{label}] wrote {json_path}")
    print(f"\n{md_content}")

    return full_results


def _get_variants(model_name: str, pt_path: Path) -> list[tuple[str, str, str, str]]:
    """Return list of (label, pt_path_str, task, export_mode) for a model name.

    yolov8 models → [("yolov8n", ..., "classic")]
    yolo26 models → [("yolo26n-classic", ..., "classic"), ("yolo26n-nmsfree", ..., "nmsfree")]
    """
    task = _infer_task(model_name)
    if _has_end2end(str(pt_path)):
        return [
            (f"{model_name}-classic", str(pt_path), task, "classic"),
            (f"{model_name}-nmsfree", str(pt_path), task, "nmsfree"),
        ]
    return [(model_name, str(pt_path), task, "classic")]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark A: Ultralytics vs yolo-validator on COCO val2017",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+", default=["yolov8n", "yolov8n-seg"],
                   metavar="MODEL",
                   help="Model names (downloads from hub if .pt not found locally).")
    p.add_argument("--max-images", type=int, default=None, metavar="N",
                   help="Limit to first N images. None = all 5000.")
    p.add_argument("--warmup", type=int, default=3, metavar="N",
                   help="Warmup frames for yolo-validator (excluded from timing).")
    p.add_argument("--output-dir", default="benchmarks/results", metavar="DIR")
    p.add_argument("--shadow-root", default="benchmarks/_coco_shadow", metavar="DIR")
    p.add_argument("--images-dir", default="~/Datasets/COCO/val2017", metavar="DIR")
    p.add_argument("--gt-json",
                   default="~/Datasets/COCO/annotations/instances_val2017.json",
                   metavar="FILE")
    p.add_argument("--device", default="cpu",
                   choices=["cpu", "cuda", "tensorrt", "coreml"],
                   help="cpu: ult-pt/ult-onnx/yv-torch/yv-numpy on CPU. "
                        "cuda: same 4 configs on GPU (Ultralytics device=0, ORT CUDAExecutionProvider). "
                        "tensorrt: ult-pt(cuda)/ult-engine/yv-tensorrt on a TRT engine. "
                        "coreml: macOS Apple Silicon — ult-onnx (FP32 anchor, CPU EP), "
                        "ult-coreml (native FP16 .mlpackage on ANE), yv-torch/yv-numpy on "
                        "the ORT CoreML EP (FP16, ANE); ult-pt skipped.")
    p.add_argument("--half", action="store_true",
                   help="FP16 precision. device=cuda: export+run a half ONNX graph "
                        "(all 4 lanes in FP16). device=tensorrt: build FP16 engines. "
                        "device=coreml always exports a FP16 .mlpackage regardless. "
                        "Not valid with device=cpu (ORT CPU has no useful FP16 path).")
    p.add_argument("--batch", type=int, default=1, metavar="N",
                   help="Inference batch size. 1 (default) = single-stream latency "
                        "reference (matches the non-batched edgefirst-profiler). N>1 "
                        "measures batched throughput, ONNX-EP only (cpu/cuda); rejected "
                        "on tensorrt/coreml (fixed batch=1).")
    p.add_argument("--skip-ult-pt", action="store_true",
                   help="Skip the Ultralytics PyTorch reference lane. Use where "
                        "the TensorRT/ONNX engine is the deployment path (e.g. "
                        "Orin): pt is redundant (pt ≡ engine) and very slow.")
    p.add_argument("--only-export-mode", choices=["classic", "nmsfree"], default=None,
                   help="Run only variants with this export mode (used by the "
                        "per-variant subprocess isolation on device=tensorrt).")
    p.add_argument("--no-isolate", action="store_true",
                   help="Disable per-variant subprocess isolation on "
                        "device=tensorrt (run all variants in one process). "
                        "Isolation prevents the pycuda (yv-tensorrt) CUDA context "
                        "from poisoning the next variant's PyTorch (ult-pt) "
                        "with CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH.")
    p.add_argument("--edgefirst-session", default=None, metavar="SESSION_ID",
                   help="EdgeFirst Studio validation session ID (e.g. v-1a8f). "
                        "When provided, fetches edgefirst-profiler accuracy + timing "
                        "from Studio and adds an edgefirst-profiler row to the "
                        "benchmark table. Intended for single-model runs; applies to "
                        "every variant when multiple --models are specified.")
    return p


def _spawn_variant_worker(model_name: str, export_mode: str, args) -> int:
    """Run a single variant in a fresh process for CUDA-context isolation.

    On device=tensorrt the yolo-validator backend uses pycuda, whose retained
    CUDA context outlives a single yv-tensorrt run and corrupts the *next*
    variant's PyTorch (ult-pt/engine export) with
    ``CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH``. Giving each variant its own
    process means the pycuda context dies cleanly on exit, so every variant
    starts from pristine CUDA state. The child inherits this process's
    environment (PYTHONIOENCODING/LANG) and working directory.
    """
    cmd = [
        sys.executable, "-m", "benchmarks.benchmark_a",
        "--models", model_name,
        "--device", args.device,
        "--only-export-mode", export_mode,
        "--warmup", str(args.warmup),
        "--output-dir", str(args.output_dir),
        "--shadow-root", str(args.shadow_root),
        "--images-dir", str(args.images_dir),
        "--gt-json", str(args.gt_json),
    ]
    if args.max_images is not None:
        cmd += ["--max-images", str(args.max_images)]
    if args.half:
        cmd.append("--half")
    if args.skip_ult_pt:
        cmd.append("--skip-ult-pt")
    if args.edgefirst_session:
        cmd += ["--edgefirst-session", args.edgefirst_session]
    print(f"[isolate] spawning fresh process: {model_name} ({export_mode})")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[isolate] worker {model_name} ({export_mode}) exited "
              f"rc={result.returncode}")
    return result.returncode


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.half and args.device == "cpu":
        parser.error("--half is not valid with --device cpu (ONNX Runtime CPU has "
                     "no useful FP16 path). Use --device cuda for FP16 ONNX, or "
                     "--device tensorrt for a FP16 engine.")

    # On device=tensorrt, isolate each variant in its own process (unless the
    # caller already narrowed to a single export mode, i.e. we ARE a worker, or
    # isolation was explicitly disabled). See _spawn_variant_worker for why.
    orchestrate = (
        args.device == "tensorrt"
        and not args.no_isolate
        and args.only_export_mode is None
    )

    if args.batch > 1 and args.device not in ("cpu", "cuda"):
        raise SystemExit(
            f"--batch {args.batch} is only supported on --device cpu|cuda (ONNX EP "
            f"dynamic batch). device={args.device} is fixed batch=1 (native CoreML / "
            f"TensorRT cannot dynamic-batch). Use --batch 1 there."
        )

    images_dir = Path(args.images_dir).expanduser()
    gt_json = Path(args.gt_json).expanduser()
    shadow_root = Path(args.shadow_root)
    output_dir = Path(args.output_dir)

    print(f"Benchmark A — models: {args.models}")
    print(f"  images_dir : {images_dir}")
    print(f"  gt_json    : {gt_json}")
    print(f"  shadow_root: {shadow_root}")
    print(f"  output_dir : {output_dir}")
    print(f"  max_images : {args.max_images}")
    print(f"  warmup     : {args.warmup}")
    print()

    from ultralytics import YOLO as _YOLO

    failed: list[tuple[str, str, str]] = []   # (label, export_mode, reason)

    for model_name in args.models:
        print(f"{'=' * 60}")
        print(f"  MODEL: {model_name}")
        print(f"{'=' * 60}")

        # Locate or download .pt
        pt_path = Path(model_name + ".pt")
        if not pt_path.exists():
            print(f"[benchmark] downloading {model_name}.pt ...")
            _YOLO(model_name + ".pt")
            if not pt_path.exists():
                import os
                hub_path = Path.home() / ".config" / "Ultralytics" / (model_name + ".pt")
                if hub_path.exists():
                    pt_path = hub_path
                else:
                    print(f"[ERROR] Could not find {model_name}.pt after download")
                    continue

        try:
            variants = _get_variants(model_name, pt_path)
        except Exception as e:
            print(f"[ERROR] Could not determine variants for {model_name}: {e}")
            continue

        for label, pt_str, task, export_mode in variants:
            if args.only_export_mode and export_mode != args.only_export_mode:
                continue
            print(f"\n{'─' * 60}")
            print(f"  VARIANT: {label} (mode={export_mode})")
            print(f"{'─' * 60}")
            if orchestrate:
                rc = _spawn_variant_worker(model_name, export_mode, args)
                if rc != 0:
                    # A subprocess that is OOM-killed (rc=-9) or crashes writes
                    # no result JSON; record it so it can't be mistaken for a
                    # variant that silently succeeded.
                    failed.append((label, export_mode, f"worker exit rc={rc}"))
                continue
            try:
                run_benchmark_a(
                    label=label,
                    pt_path=pt_str,
                    task=task,
                    export_mode=export_mode,
                    shadow_root=shadow_root,
                    images_dir=images_dir,
                    gt_json=gt_json,
                    output_dir=output_dir,
                    max_images=args.max_images,
                    warmup=args.warmup,
                    device=args.device,
                    half=args.half,
                    skip_ult_pt=args.skip_ult_pt,
                    edgefirst_session_id=args.edgefirst_session,
                    batch=args.batch,
                )
            except Exception as e:
                print(f"[ERROR] {label} failed: {e}")
                import traceback
                traceback.print_exc()
                failed.append((label, export_mode, str(e)))

    if failed:
        print(f"\n{'=' * 60}\n  {len(failed)} VARIANT(S) FAILED\n{'=' * 60}")
        for label, mode, reason in failed:
            print(f"  - {label} ({mode}): {reason}")
        raise SystemExit(1)
    print("\nAll variants completed.")


if __name__ == "__main__":
    main()
