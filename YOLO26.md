# YOLO26 end-to-end (NMS-free) vs classical-NMS under FP32 / FP16 / INT8

This document characterizes the YOLO26 detection head in its two forms — the native end-to-end (NMS-free) head and the classical anchor-grid head that needs class-aware NMS — across FP32, FP16, and the INT8 quantization variants, measured locally on the full COCO val2017 (5000 images). It documents what works, what does not, and the measured accuracy (mAP) and latency. The bottom line for edge deployment: the classical head is more accurate at every precision, no slower, and quantizes far better, so EdgeFirst standardizes on the classical (`end2end=False`) head — this document is the evidence behind that choice.

## Scope and platform

- CPU: 13th Gen Intel Core i9-13900 (x86_64); GPU: NVIDIA RTX 4060 (8 GiB).
- TFLite artifacts run on the host CPU via the TensorFlow Lite / LiteRT interpreter (XNNPACK), single-stream (batch 1), imgsz 640.
- Model: yolo26n (detection). The procedure extends to s/m and to the segmentation heads as follow-up.
- Export environment: Ultralytics 8.4.75, onnx2tf 1.28.8, TensorFlow 2.21.0 (isolated `venv-tfexport`). Full-integer / integer_quant INT8 calibration: 500 COCO train2017 images (seeded, disjoint from val2017). Dynamic-range INT8 built from the saved_model with `Optimize.DEFAULT` (no representative data).
- ONNX FP32 (CPU/CUDA) and FP16 (CUDA) reference rows live in `benchmarks/metrics/onnx-cpu.json` and `benchmarks/metrics/onnx-cuda.json`.

## Methodology note (measurement validity)

The end-to-end head emits one `[1, 300, 6]` tensor `[x1, y1, x2, y2, score, class]` (boxes normalized in letterboxed space, already NMS-free / top-k). The classical head emits the raw `[1, 84, 8400]` anchor grid that still needs class-aware NMS.

Decoder trust was established before any number was accepted:

- Ultralytics' own `yolo val` decodes **float-IO** TFLite correctly (end-to-end FP32 reproduced the known 0.397) but **mis-decodes full-integer int8-IO** onnx2tf artifacts — it reported the classical INT8 model at 0.00003 where the trusted path gives 0.351 (a ~10000x error). `yolo val` int8-IO readings are discarded.
- The repo's `tflite_infer.py` path (correct dequantization, normalized-box handling, pycocotools "crowd-as-normal" scoring — identical to every other row in this repo) is the trusted decoder. It was extended with an end-to-end `[1,300,6]` decode branch (detect) and re-validated against the FP32 anchor (end-to-end TFLite FP32 = 0.3963 vs ONNX 0.3962) before any INT8 cell was trusted, so every result here is reproducible from the repo.

Every cell below: trusted decoder, pycocotools crowd-as-normal, conf 0.001, iou 0.7, max_det 300.

## What upstream export actually supports (proven, not assumed)

- Upstream Ultralytics + onnx2tf **does** export the native end-to-end (NMS-free) head all the way to a full-integer INT8 TFLite — the `[1,300,6]` output survives, and a `*_full_integer_quant.tflite` with int8 input AND int8 output is produced. The prior pipeline that forced `end2end=False` was a limitation of our decoder, not of upstream export.
- The full-integer end-to-end graph keeps 8 tensors float (the NMS-free top-k / selection ops); the rest is int8.
- The end-to-end head does NOT survive int8 **activation** quantization. `integer_quant` (full int8 compute, float IO) and `full_integer_quant` (int8 IO) collapse to the identical box AP 0.0113 with the identical 30,453 detections — the IO dtype is irrelevant; the damage is in the int8 activations of the NMS-free path. The decisive control: dynamic-range quant (int8 weights, float activations) holds 0.3922 — essentially FP32. So the head tolerates int8 weights; full-activation PTQ is what destroys it.
- The classical head survives the same full-activation PTQ at box AP 0.3514 (−5.1 pp from FP32).

## Accuracy — yolo26n detection, full val2017 (box AP / AP50)

| precision / quant | TFLite variant | IO | END2END AP | END2END AP50 | CLASSIC AP | CLASSIC AP50 |
|---|---|---|---|---|---|---|
| FP32 | float32 | float | 0.3963 | 0.5487 | 0.4024 | 0.5587 |
| FP16 | float16 | float | 0.3933 | 0.5445 | 0.4023 | 0.5587 |
| INT8 dynamic-range (default int8: int8 weights, float compute) | dynamic_range_quant | float | 0.3922 | 0.5458 | 0.3998 | 0.5559 |
| INT8 "with float ops" (full int8 compute, float IO) | integer_quant | float IO | 0.0113 | 0.0415 | 0.3514 | 0.5186 |
| INT8 full integer (NPU-grade) | full_integer_quant | int8 IO | 0.0113 | 0.0415 | 0.3514 | 0.5186 |

Per-head INT8 drop from FP32: classical −0.3 pp (dynamic) / −5.1 pp (full integer); end-to-end −0.4 pp (dynamic) / −38.5 pp (full integer, a 97% loss).

## Performance — yolo26n detection, host CPU single-stream (XNNPACK)

Per-image inference (ms) and wall FPS, same runs.

| precision / quant | END2END inf ms | END2END FPS | CLASSIC inf ms | CLASSIC FPS |
|---|---|---|---|---|
| FP32 | 46.65 | 19.9 | 46.48 | 19.2 |
| FP16 | 46.39 | 20.4 | 46.42 | 19.5 |
| INT8 dynamic-range | 30.61 | 29.2 | 30.69 | 27.9 |
| INT8 float-IO | 25.92 | 35.4* | 25.54 | 33.0 |
| INT8 full integer | 25.59 | 35.1* | 25.32 | 32.4 |

\* End-to-end INT8 float-IO/full-integer FPS is inflated by the accuracy collapse (only 30k vs 667k detections → near-zero postprocess); it is not a real throughput advantage. At matched (working) precisions the two heads are within ~1% on CPU inference, and the classical head's class-aware NMS postprocess is only ~0.3–2.8 ms.

FP16 on CPU shows no speedup (TFLite upcasts float16 to float32) — accuracy ≈ FP32, latency ≈ FP32. INT8 (any working form) is the real CPU speedup (~1.5x faster than FP32). TFLite does not exercise the GPU; the ONNX CPU/CUDA FP32/FP16 rows corroborate that the two heads run at parity in latency.

## What works / what does not (yolo26n, local x86_64)

- end-to-end FP32 / FP16 TFLite: works (0.396 / 0.393).
- end-to-end INT8 dynamic-range (default int8): works — 0.392, near-FP32, and ~1.5x faster on CPU.
- end-to-end INT8 float-IO (integer_quant): broken — 0.011.
- end-to-end INT8 full-integer (int8 IO, NPU-grade): broken — 0.011.
- classical FP32 / FP16 / INT8 dynamic-range: works (0.402 / 0.402 / 0.400).
- classical INT8 float-IO and full-integer (NPU-grade): works — 0.351 (−5.1 pp).

Practical takeaway for edge: NPU delegates require full-integer (int8-activation) models. In that regime the classical head is deployable (0.351) and the end-to-end head is not (0.011). Where float activations are allowed (CPU/GPU dynamic-range), both heads are viable and within ~0.8 pp.

## Conclusion

For edge deployment the end-to-end (NMS-free) head offers no benefit over the classical head, and a clear disadvantage under quantization:

- **Accuracy:** classical ≥ end-to-end at every precision — marginally at FP32/FP16/dynamic-range (+0.6–0.9 pp), decisively under full-activation INT8 (+34 pp; classical 0.351 vs end-to-end 0.011).
- **Quantization:** the regime edge NPU delegates require — full-integer (int8-activation) — is exactly where the end-to-end head collapses (97% AP loss) and the classical head holds (−5 pp). The classical head quantizes far better.
- **Performance:** at matched, working precisions the two heads are within ~1% on CPU inference; the classical head's class-aware NMS adds only ~0.3–2.8 ms, so it is no slower in practice on the edge targets.

This justifies EdgeFirst standardizing on the classical (`end2end=False`) head for edge. This study scopes CPU/TFLite on the edge-relevant targets; the one remaining open question — whether end-to-end's NMS-free design wins on **high-throughput GPU server** inference, where NMS is a batched synchronization bottleneck — is outside the edge focus and not evaluated here.

## Reproduction

The TFLite artifacts are under `benchmarks/results/_models_tflite/` (`yolo26n-e2e-*.tflite`, `yolo26n-classic-*.tflite`). Each is scored with `benchmarks/tflite_infer.py` (`--delegate none` on the host) which now decodes both the classical `[1,84,8400]` and the end-to-end `[1,300,6]` outputs. Export is reproduced by `yolo export format=tflite int8=True data=<calib>` with the native head (end-to-end) or with `Detect.end2end=False` (classical), calibrated on 500 train2017 images.
