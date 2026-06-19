# Benchmark — EdgeFirst stack vs the reference validator

This document measures the **EdgeFirst stack** against a reference validator across platforms, and then establishes the **yolo-validator** proxy that provides that reference where the Ultralytics validator cannot run. The reference is the **Ultralytics validator** where it runs (CPU, CUDA, Jetson); on platforms without PyTorch/CUDA (Hailo, Ara2, i.MX) the validated yolo-validator proxy stands in.

All accuracy is COCO val2017 (5000 images), one canonical pycocotools re-score per config. **iscrowd filtering is not currently supported** — every annotation is scored as a normal target, matching how the EdgeFirst stack validates, so the EdgeFirst-vs-reference deltas below are convention-matched. Host: RTX 4060, i9-13900F, Ubuntu 24.04, torch 2.6.0+cu124 / ultralytics 8.4.68 / onnxruntime-gpu 1.21.0. Parity: `conf=0.001 iou=0.7 max_det=300 imgsz=640 rect=False batch=1`.

---

# Part 1 — EdgeFirst vs the reference (RTX 4060)

The edgefirst-profiler is a pipelined runner (4× in-flight) on the same byte-identical v8.4.0 ONNX checkpoint as the reference. EdgeFirst AP columns are FP32 (matched to the FP32 reference); EdgeFirst FPS and speedup are the FP16 deployment runs (FP16 ≈ FP32 accuracy, Δ ≤ 0.05 pp, ~30% faster). speedup = EdgeFirst FPS ÷ Ultralytics FPS<sub>wall</sub>.

## Detection

| Variant | reference box | EdgeFirst box | **EdgeFirst Δ** | ref FPS | EdgeFirst FPS | speedup |
|---|--:|--:|--:|--:|--:|--:|
| yolov5nu | 0.3371 | 0.3295 | **−0.0076** | 137 | 452 | **3.3×** |
| yolov5su | 0.4219 | 0.4126 | **−0.0093** | 110 | 318 | 2.9× |
| yolov5mu | 0.4808 | 0.4699 | **−0.0109** | 68 | 167 | 2.5× |
| yolov8n | 0.3671 | 0.3583 | **−0.0088** | 132 | 448 | 3.4× |
| yolov8s | 0.4425 | 0.4322 | **−0.0103** | 99 | 296 | 3.0× |
| yolov8m | 0.4943 | 0.4845 | **−0.0098** | 56 | 148 | 2.7× |
| yolo11n | 0.3867 | 0.3784 | **−0.0083** | 132 | 440 | 3.3× |
| yolo11s | 0.4587 | 0.4488 | **−0.0099** | 100 | 304 | 3.0× |
| yolo11m | 0.5051 | 0.4959 | **−0.0092** | 60 | 152 | 2.5× |
| yolo26n-classic | 0.4022 | 0.3969 | **−0.0053** | 136 | 431 | 3.2× |
| yolo26n-nmsfree | 0.3963 | — | — | 138 | — | — |
| yolo26s-classic | 0.4774 | 0.4712 | **−0.0062** | 100 | 296 | 2.9× |
| yolo26s-nmsfree | 0.4695 | — | — | 100 | — | — |
| yolo26m-classic | 0.5240 | 0.5187 | **−0.0053** | 59 | 156 | 2.6× |
| yolo26m-nmsfree | 0.5164 | — | — | 60 | — | — |

EdgeFirst runs **2.5–3.4×** the reference throughput at a **0.5–1.1 pp** box-AP cost. EdgeFirst publishes one decode per model (the classic head), so the NMS-free rows have no EdgeFirst lane.

## Segmentation

| Variant | ref box | EdgeFirst box | **box Δ** | ref mask | EdgeFirst mask | **mask Δ** | ref FPS | EdgeFirst FPS | speedup |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| yolov8n-seg | 0.3603 | 0.3544 | **−0.0059** | 0.3020 | 0.2868 | **−0.0152** | 20.5 | 333 | **16.3×** |
| yolov8s-seg | 0.4392 | 0.4324 | **−0.0068** | 0.3635 | 0.3437 | **−0.0198** | 23.4 | 222 | 9.5× |
| yolov8m-seg | 0.4896 | 0.4810 | **−0.0086** | 0.4028 | 0.3764 | **−0.0264** | 20.6 | 119 | 5.8× |
| yolo11n-seg | 0.3827 | 0.3740 | **−0.0087** | 0.3190 | 0.3016 | **−0.0174** | 22.8 | 320 | 14.0× |
| yolo11s-seg | 0.4558 | 0.4477 | **−0.0081** | 0.3744 | 0.3527 | **−0.0217** | 24.5 | 225 | 9.2× |
| yolo11m-seg | 0.5051 | 0.4984 | **−0.0067** | 0.4139 | — | — | 22.3 | 152 | 6.8× |
| yolo26n-seg-classic | 0.3993 | 0.3946 | **−0.0047** | 0.3408 | 0.3271 | **−0.0137** | 24.6 | 290 | 11.8× |
| yolo26n-seg-nmsfree | 0.3901 | — | — | 0.3343 | — | — | 25.3 | — | — |
| yolo26s-seg-classic | 0.4730 | 0.4686 | **−0.0044** | 0.4000 | 0.3816 | **−0.0184** | 24.7 | 207 | 8.4× |
| yolo26s-seg-nmsfree | 0.4660 | — | — | 0.3952 | — | — | 25.7 | — | — |
| yolo26m-seg-classic | 0.5231 | 0.5204 | **−0.0027** | 0.4397 | 0.4200 | **−0.0197** | 20.6 | 110 | 5.4× |
| yolo26m-seg-nmsfree | 0.5138 | — | — | 0.4336 | — | — | 20.6 | — | — |

Segmentation is where the pipeline gain is largest: **5.4–16.3×** the single-stream reference, because the reference is bottlenecked on the mask postprocess EdgeFirst overlaps across workers. Box Δ is **0.3–0.9 pp**; the mask Δ is larger (**1.4–2.6 pp**) — the residual is the EdgeFirst mask decode/export pipeline, not characterized further here.

## Per-stage latency (trio)

`latency` is the per-frame stage sum; references are single-stream (throughput ≈ 1000/latency minus I/O), while the edgefirst-profiler overlaps up to 4 frames so its FPS far exceeds 1000/latency at comparable per-frame latency. EdgeFirst `pre` includes JPEG decode + H2D, which the references exclude.

| Variant | config | pre | inf | post | latency (ms) | FPS |
|---|---|--:|--:|--:|--:|--:|
| yolov8n | reference (ONNX) | 0.3 | 3.2 | 0.4 | 3.9 | 132 |
| yolov8n | edgefirst-profiler | 7.8 | 6.3 | 0.8 | 11.9 | **448** |
| yolov8n-seg | reference (ONNX) | 0.3 | 3.9 | 10.6 | 14.8 | 20.5 |
| yolov8n-seg | edgefirst-profiler | 9.0 | 7.9 | 4.2 | 17.9 | **333** |
| yolo26n-classic | reference (ONNX) | 0.3 | 3.2 | 0.4 | 3.9 | 136 |
| yolo26n-classic | edgefirst-profiler | 7.6 | 6.4 | 0.7 | 11.8 | **431** |

---

# Part 2 — proxy fidelity: yolo-validator vs the Ultralytics validator

This establishes yolo-validator as a faithful stand-in for the reference, so the EdgeFirst comparison holds on platforms where the Ultralytics validator cannot run. Δ = yolo-validator (NumPy path) − Ultralytics validator; the Torch path gives the identical Δ.

## Accuracy — full CUDA breadth (27 variants)

**Detection**

| Variant | ult box | **yv box Δ** | | Variant | ult box | **yv box Δ** |
|---|--:|--:|---|---|--:|--:|
| yolov5nu | 0.3371 | **−0.0030** | | yolo11m | 0.5051 | **−0.0022** |
| yolov5su | 0.4219 | **−0.0017** | | yolo26n-classic | 0.4022 | **−0.0034** |
| yolov5mu | 0.4808 | **−0.0008** | | yolo26n-nmsfree | 0.3963 | **−0.0046** |
| yolov8n | 0.3671 | **−0.0020** | | yolo26s-classic | 0.4774 | **−0.0022** |
| yolov8s | 0.4425 | **−0.0033** | | yolo26s-nmsfree | 0.4695 | **−0.0040** |
| yolov8m | 0.4943 | **−0.0016** | | yolo26m-classic | 0.5240 | **−0.0019** |
| yolo11n | 0.3867 | **−0.0049** | | yolo26m-nmsfree | 0.5164 | **−0.0010** |
| yolo11s | 0.4587 | **−0.0021** | | | | |

**Segmentation**

| Variant | ult box | **yv box Δ** | ult mask | **yv mask Δ** |
|---|--:|--:|--:|--:|
| yolov8n-seg | 0.3603 | **−0.0017** | 0.3020 | **−0.0037** |
| yolov8s-seg | 0.4392 | **−0.0009** | 0.3635 | **−0.0036** |
| yolov8m-seg | 0.4896 | **−0.0015** | 0.4028 | **−0.0048** |
| yolo11n-seg | 0.3827 | **−0.0043** | 0.3190 | **−0.0060** |
| yolo11s-seg | 0.4558 | **−0.0035** | 0.3744 | **−0.0061** |
| yolo11m-seg | 0.5051 | **−0.0022** | 0.4139 | **−0.0061** |
| yolo26n-seg-classic | 0.3993 | **−0.0042** | 0.3408 | **−0.0053** |
| yolo26n-seg-nmsfree | 0.3901 | **−0.0040** | 0.3343 | **−0.0048** |
| yolo26s-seg-classic | 0.4730 | **−0.0032** | 0.4000 | **−0.0050** |
| yolo26s-seg-nmsfree | 0.4660 | **−0.0036** | 0.3952 | **−0.0055** |
| yolo26m-seg-classic | 0.5231 | **−0.0014** | 0.4397 | **−0.0037** |
| yolo26m-seg-nmsfree | 0.5138 | **−0.0011** | 0.4336 | **−0.0033** |

**Envelope: ≤ 0.49 pp box / ≤ 0.61 pp mask**, one-sided (yolo-validator always slightly lower) from letterbox-pad rounding and greedy-vs-torchvision NMS tie-breaking; the mask-materialization stage is bit-identical given identical input boxes. yolo-validator Torch path ≡ NumPy path to ≥ 4 decimals; Ultralytics `.pt` ≡ `.onnx` to ≤ 0.0001 box.

## Performance — CPU FP32 (the honest no-CUDA comparison)

The CUDA comparison is not apples-to-apples for performance: there the Ultralytics validator offloads NMS + mask to the GPU via torch, while the portable path runs them on the CPU. The platforms yolo-validator actually serves have no torch/CUDA, so this is the representative test — both on CPU, same hardware (RTX 4060 host, `device=cpu`).

| Variant | ult box | yv box Δ | ult mask | yv mask Δ | ult FPS | yv FPS |
|---|--:|--:|--:|--:|--:|--:|
| yolov5nu | 0.3371 | −0.0030 | — | — | 37.9 | 32.3 |
| yolov8n | 0.3670 | −0.0019 | — | — | 36.9 | 31.8 |
| yolo11n | 0.3866 | −0.0048 | — | — | 37.2 | 35.5 |
| yolo26n-classic | 0.4022 | −0.0033 | — | — | 45.3 | 43.6 |
| yolo26n-nmsfree | 0.3962 | −0.0046 | — | — | 46.6 | 54.0 |
| yolov8n-seg | 0.3604 | −0.0016 | 0.3018 | −0.0034 | 3.5 | 6.5 |
| yolo11n-seg | 0.3826 | −0.0042 | 0.3189 | −0.0058 | 3.9 | 6.7 |
| yolo26n-seg-classic | 0.3992 | −0.0042 | 0.3408 | −0.0053 | 4.3 | 7.6 |
| yolo26n-seg-nmsfree | 0.3901 | −0.0039 | 0.3342 | −0.0048 | 4.3 | 8.3 |

On CPU yolo-validator tracks the Ultralytics validator within ~15% on detection, and is faster on segmentation (Ultralytics' CPU `process_mask_native` is heavier than the portable NumPy mask path). It is a faithful proxy in both accuracy and performance on the hardware class it targets.

---

# Platform expansion (next)

Nano coverage is the baseline; small/medium and additional platforms follow. The EdgeFirst stack already publishes results on Hailo-8L, NXP i.MX (8M Plus / 95 Neutron / Ara2), Apple CoreML, and Jetson Orin Nano — on those the validated yolo-validator proxy is the reference (local proxy runs to come). One convention-independent proxy advantage already measured on the Jetson Orin Nano (7.4 GiB): the Ultralytics validator OOM-kills on 5000-image segmentation (it accumulates the whole dataset's predictions, ~4.2 GiB), while yolo-validator RLE-encodes each mask and streams (peak 3.0 GiB, never swaps).

---
*COCO val2017 (5000 images), detection + segmentation, crowd-as-normal eval (iscrowd not filtered). Part 1: EdgeFirst vs the Ultralytics-validator reference on RTX 4060 (FP32 accuracy, FP16 deployment throughput). Part 2: yolo-validator proxy fidelity — ≤ 0.49 pp box / ≤ 0.61 pp mask vs the reference across 27 variants, with the honest performance comparison on CPU.*
