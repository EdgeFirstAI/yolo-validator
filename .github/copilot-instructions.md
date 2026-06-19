# Copilot Instructions

Standalone, portable reference validator for Ultralytics YOLO detection and segmentation across runtimes (ONNX Runtime and TensorRT implemented; TFLite/HailoRT/Ara2 planned). It reproduces the Ultralytics validation pipeline using the same OpenCV/NumPy operations as Ultralytics, with no hard PyTorch/CUDA dependency, so it serves as the accuracy and performance reference on platforms the Ultralytics validator cannot run (Hailo, Ara2, i.MX, plain CPU). It is the proxy used to benchmark the EdgeFirst stack.

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
- **iscrowd filtering is not currently supported** — `canonical_eval` scores every annotation as a normal target (crowd-as-normal), matching how the EdgeFirst stack validates.
- Results and persisted predictions go under `benchmarks/results/` (gitignored, on-disk) — never `/tmp`. `benchmark_a.py` auto-expands yolo26/yolov10 into classic + nmsfree variants.
- See `BENCHMARK.md` (Part 1: EdgeFirst vs reference; Part 2: proxy fidelity) and `README.md`.

## Scope
ONNX Runtime and TensorRT backends are implemented. TFLite/HailoRT/Ara2 land behind the `Backend` protocol in `yolo_validator/backends/`.
