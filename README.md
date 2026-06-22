# yolo-validator

**yolo-validator** is a portable, Apache-2.0 proxy for the Ultralytics YOLO validator. It reproduces the Ultralytics detection and segmentation validation pipeline using the same OpenCV/NumPy operations Ultralytics uses, but with no hard dependency on PyTorch or CUDA — so it serves as the accuracy and performance reference on the many platforms the Ultralytics validator cannot reach (Hailo, NXP i.MX and Ara2, Apple CoreML, NVIDIA Jetson, plain CPU), with memory-frugal streaming that avoids the out-of-memory failures the Ultralytics validator hits on constrained edge hardware.

Its purpose is to validate the EdgeFirst stack consistently across all of those targets: where the Ultralytics validator runs, it is the reference; where it cannot, yolo-validator stands in. yolo-validator reproduces the Ultralytics validator within sub-1 pp box and mask mAP wherever both run — the proxy-fidelity comparison is at the end of **[BENCHMARK.md](BENCHMARK.md)**, and the EdgeFirst-vs-reference results across platforms are at the top.

Two interchangeable preprocess/postprocess paths back this up:
- **Torch path** — reuses Ultralytics ops (letterbox, NMS) for numerically-faithful predictions where PyTorch is available.
- **NumPy/OpenCV path** — the portable path that runs on edge silicon without PyTorch/CUDA.

Mask materialization is shared (NumPy) by both paths and ports the Ultralytics native mask pipeline (`process_mask_native` + `scale_masks` + `crop_mask`); it is bit-identical to Ultralytics given identical input boxes.

## Install

| Path | Install | Notes |
|---|---|---|
| Core (NumPy/OpenCV) | `pip install yolo-validator` | No ONNX/Torch; AGPL-free |
| ONNX Runtime | `pip install 'yolo-validator[onnx]'` | CPU; `--provider cuda\|coreml` for GPU |
| TensorRT | platform-provided `tensorrt` + `pycuda` | Jetson/NVIDIA only |
| Torch (Ultralytics-faithful) | `pip install 'yolo-validator[torch]'` | Pulls Ultralytics (**AGPL-3.0**) |

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e '.[onnx,dev]'
```

## Usage

```bash
# Detection or segmentation over an image dir, scored against COCO ground truth.
python -m yolo_validator --model yolov8n-seg.onnx --images ./images/ \
    --gt instances_val2017.json --task segment

# TensorRT backend (Jetson — requires platform tensorrt + pycuda):
python -m yolo_validator --model yolo26n.engine --runtime tensorrt \
    --images ./images/ --gt instances_val2017.json
```

The `--gt` `image_id` is the integer filename stem; non-numeric names fall back to a CRC32 that will not match COCO ground-truth ids (fine for COCO `val2017`).

## Status

v0.2.0 — NumPy/OpenCV, ONNX Runtime, and TensorRT backends, with HailoRT detection and instance segmentation measured on-target (see [BENCHMARK.md](BENCHMARK.md)). New edge backends plug in behind the `Backend` protocol.

## License

yolo-validator is licensed under the **Apache License 2.0** (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).

It is developed independently of and is not affiliated with Ultralytics. **No Ultralytics source code is copied or vendored here.** Ultralytics (licensed **AGPL-3.0**) is an *optional* dependency, pulled only by the `[torch]` extra to provide a numerically-faithful Torch reference path and the benchmark harness. The core NumPy/OpenCV path and the ONNX Runtime backend — the paths intended for edge silicon — carry **no AGPL obligation**. COCO val2017 is CC BY 4.0; pycocotools is BSD-style. See [NOTICE](NOTICE) for attributions.
