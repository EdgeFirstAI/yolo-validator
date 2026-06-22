# Copilot Instructions

Standalone, portable reference validator for Ultralytics YOLO detection and segmentation across runtimes (ONNX Runtime — CPU/CUDA/CoreML providers — and TensorRT implemented as core backends; HailoRT INT8 measured on-target via the benchmark harness; TFLite/Ara2 planned). It reproduces the Ultralytics validation pipeline using the same OpenCV/NumPy operations as Ultralytics, with no hard PyTorch/CUDA dependency, so it serves as the accuracy and performance reference on platforms the Ultralytics validator cannot run (Hailo, Ara2, i.MX, plain CPU). It is the proxy used to benchmark the EdgeFirst stack.

## Commands
```bash
source venv/bin/activate
pytest                       # full suite (hermetic; ONNX test skips w/o model)
python -m yolo_validator --help
# Benchmark harness (device = cpu | cuda | tensorrt):
python -m benchmarks.benchmark_a --models yolov8n yolov8n-seg yolo26n --device cpu \
    --images-dir ~/coco/val2017 --gt-json ~/coco/annotations/instances_val2017.json \
    --output-dir benchmarks/results/run
```

## Invariants
- **Single-stream by design.** One frame at a time, every stage to completion, so per-stage timing is additive — never pipeline or overlap stages. Pipelining is the EdgeFirst profiler's story, not this tool's; the goals here are portability and low peak memory (frame-by-frame streaming with immediate per-mask RLE encode), not raw speed.
- **Two equivalent paths.** The Torch path (reuses Ultralytics letterbox/NMS) and the portable NumPy/OpenCV path must produce the same detections within float tolerance (≥ 4 decimals); only timing differs. load/decode is shared OpenCV code. The NumPy path is the edge path that runs without torch/CUDA.
- **Mask materialization** is shared NumPy (`process_mask_native` + `scale_masks` + `crop_mask`) for both paths, bit-identical to Ultralytics given identical input boxes.
- **Stats:** min/mean/p50/p95/p99/max, NO trimming.
- **Warmup** runs real images through the full pipeline and is excluded from measurement. Default 3.
- **COCO defaults match Ultralytics val:** conf=0.001, iou=0.7, max_det=300.

## Benchmarks (`benchmarks/`)
- Three lanes: Ultralytics validator (reference) → yolo-validator (proxy) → edgefirst-profiler. One canonical pycocotools re-score per config (`canonical_eval.py`).
- **EdgeFirst comparison** (`studio_fetch.py` + `compare_edgefirst.py`): `studio_fetch.py` is the live Studio JSON-RPC client — `fetch_studio_metrics(session_id)` / `edgefirst_canonical(session_id)` return AP as fractions plus on-target timing/fps (det from `detection.summary`, seg from `segmentation.summary`, timing from `timing.inline`). `compare_edgefirst.py --catalog <export>` joins EdgeFirst sessions against our `metrics/<platform>.json` rows and prints per-model and gap-summary tables; it fetches each session via `studio_fetch` (cached under `results/`). The `--catalog` file is the EdgeFirst Studio metrics export (`edgefirst-model-zoo-metrics.json`) — a flat per-session record schema (AP in **percentage points 0–100**: `det_*`/`mask_*` full 12-metric summaries; `e2e_latency_ms`/`latency_mean_ms`; `realized_fps_scalar`; `is_reference`/`flag`/`delta_ap50`/`delta_mask_ap`; `session_id`/`session_url`); used ONLY as the session catalog (join keys + `session_id`), never hardcode its path. Join: EdgeFirst `version`+`size`(+`-seg`) → our `model` (`yolov5*`→`yolov5*u`, the anchor-free retrain only); `det/seg`→`detect/segment`; platform/precision verbatim.
- **Accuracy is the comparable axis; latency/fps are NOT** — this tool is single-stream (additive per-stage timing), the EdgeFirst profiler pipelines/overlaps stages. Primary metric: detection = box mAP@0.5:0.95 (and AP50), segmentation = mask mAP@0.5:0.95.
- **iscrowd filtering is not currently supported** — `canonical_eval` scores every annotation as a normal target (crowd-as-normal), matching how the EdgeFirst stack validates.
- Results and persisted predictions go under `benchmarks/results/` (gitignored, on-disk) — never `/tmp`. `benchmark_a.py` auto-expands yolo26/yolov10 into classic + nmsfree variants.
- **Doc hygiene — published vs internal.** `README.md` and `BENCHMARK.md` are published: they carry only settled, defensible results and never internal status, TODOs, "in progress"/"to come" notes, or claims of incomplete/poor EdgeFirst performance. The one acknowledged published limitation is the missing NMS-free yolo26 lane (we run the classic-NMS variant only; `-nmsfree` rows show `—`). All in-flight tracking — the platform/lane coverage matrix, open gaps, and any EdgeFirst result still pending analysis — lives in `STATUS.md` (gitignored, internal). When a measurement looks like an EdgeFirst regression (slower than, or no faster than, the reference; an unexplained accuracy gap), record it in `STATUS.md` for analysis; do not assert it in the published docs until it is confirmed and corrected.
- See `BENCHMARK.md` (Part 1: EdgeFirst vs reference; Part 2: proxy fidelity) and `README.md`.

## Scope
Core backends behind the `Backend` protocol (`yolo_validator/backends/`, via `load_backend`): ONNX Runtime (`--provider cpu|cuda|coreml`) and TensorRT. HailoRT INT8 is exercised on-target by `benchmarks/hailo_infer.py` (shared `canonical_eval`), producing the `rpi5-hailo8l` results — a benchmark-side on-device path, not a registered core backend. TFLite and Ara2 (NXP i.MX / Ara240 NPU) are planned, to land behind the same `Backend` protocol.
