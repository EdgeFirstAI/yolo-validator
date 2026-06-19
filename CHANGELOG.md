# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow SemVer.

## [Unreleased]

## [0.2.0] - 2026-06-15
### Added
- TensorRT backend (TRT 10.x + pycuda) behind the `Backend` protocol, with
  CUDA-event device timing; direct on-device comparison vs the Ultralytics
  validator on Jetson.
- Benchmark A harness: Ultralytics vs yolo-validator on COCO val2017 with
  canonical pycocotools re-scoring and stage re-binning; NMS-free vs classic
  yolo26 support; per-variant subprocess isolation on `--device tensorrt`.
- `--e2e/--no-e2e` override and a robust NMS-free auto-detection.
- Apache-2.0 release collateral: `LICENSE` copyright, `NOTICE`,
  `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`; PyPI metadata
  (SPDX license, classifiers); coverage gate in CI.

### Fixed
- **Segmentation masks** now faithfully port Ultralytics `process_mask_native`
  (raw-logit threshold, input-resolution crop with letterboxed boxes, single
  pad-strip resize) — verified **bit-identical** to Ultralytics
  (`tests/test_masks_fidelity.py`). Removes the prior systematic mask-mAP
  offset.
- **Box un-letterboxing** uses Ultralytics' rounded integer pad (matches
  `scale_boxes` and the preprocess pad placement).
- Mask-coefficient count read from the proto shape instead of a hardcoded 32.
- Benchmark timing: warmup frames no longer contaminate per-stage stats;
  measured-loop wall time enables apples-to-apples wall-clock FPS; FPS computed
  in code (wall + pipeline definitions); device CUDA-event timings surfaced
  through rebin.
- Robustness: subprocess-worker failures (incl. OOM) are surfaced and exit
  non-zero; UTF-8 file writes throughout; pycocotools `-1.0` sentinel rendered
  as N/A; deterministic CRC32 `image_id` fallback; CLI guards `--max-images`
  ≤ `--warmup`; markdown table separator always matches the header.

### Changed
- `requires-python` documented; version bumped to 0.2.0.

## [0.1.0] - 2026-06-14
### Added
- Standalone YOLO detection+segmentation validator core.
- ONNX Runtime backend behind a pluggable `Backend` protocol.
- Dual preprocess/postprocess paths (Torch via Ultralytics ops; NumPy/OpenCV fallback).
- Serial per-stage timing (min/mean/p50/p95/p99/max, no trim) with measurement-excluded warmup.
- pycocotools COCO evaluation (bbox + segm) and COCO JSON output.
- CLI mirroring Ultralytics `val` (conf=0.001, iou=0.7, max_det=300).
