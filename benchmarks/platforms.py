"""Platform registry and canonical metrics schema for cross-platform benchmarks.

This repo measures the **Ultralytics/Vendor baseline**: models are converted and
quantized with Ultralytics' own integration workflows (and any silicon-vendor
steps on top), and validated with the Ultralytics validator where it runs.
**yolo-validator fills the gaps** — it produces on-target COCO mAP where the
Ultralytics/vendor workflow cannot (e.g. HailoRT, NXP NPU, Ara2). EdgeFirst
Studio is a separate optimized lane, driven by EdgeFirst tooling and integrated
for the comparison.

Metrics for each platform are normalized into one committed file
``benchmarks/metrics/<platform>.json`` (one file per platform → parallel
per-platform branches merge with zero conflicts). ``aggregate.py`` reads them
all to regenerate BENCHMARK.md.

Canonical metrics schema (``benchmarks/metrics/<platform>.json``)::

    {
      "platform": "<id>",                # EdgeFirst-aligned id, see PLATFORMS
      "host": { ... },                   # host meta captured by the run
      "eval": "crowd-as-normal",         # iscrowd not filtered
      "rows": [
        {
          "model":     "yolov8n",        # base model (mode suffix stripped)
          "variant":   "yolov8n",        # label incl. -classic/-nmsfree/-seg
          "task":      "detect",         # detect | segment
          "precision": "FP32",           # FP32 | FP16 | INT8 | INT16
          "quant": {                     # how the artifact was quantized
            "method": "none",            # none | trt-entropy | hailo-model-zoo
                                         #      | tflite-ptq | nxp-eiq | ...
            "calib":  null               # e.g. "train2017/500" | null
          },
          "lane":      "baseline",       # baseline | edgefirst
          "validator": "ultralytics",    # ultralytics | yolo-validator
                                         #      | edgefirst-studio
          "workflow":  "ultralytics",    # ultralytics | vendor:<name>
                                         #      | edgefirst-studio
          "engine":    "onnx",           # pytorch|onnx|tensorrt|coreml|hailo|...
          "box_ap": 0.0, "box_ap50": 0.0,
          "mask_ap": null, "mask_ap50": null,
          "fps_wall": 0.0,
          "latency_ms": {"pre": 0.0, "inf": 0.0, "post": 0.0, "e2e": 0.0},
          "n_images": 5000
        }
      ]
    }
"""
from __future__ import annotations

# Nano-first standard model set — every platform runs the same set, expand to
# small/medium later. yolo26* auto-expand into -classic and -nmsfree variants.
STANDARD_MODELS = {
    "detect": ["yolov5nu", "yolov8n", "yolo11n", "yolo26n"],
    "segment": ["yolov8n-seg", "yolo11n-seg", "yolo26n-seg"],
}

# Platform registry. ``baseline_validator`` is who measures the baseline lane:
# "ultralytics" where the Ultralytics validator runs on-target, else
# "yolo-validator" (the gap-filler). ``vendor`` names any silicon-vendor
# conversion/quantization workflow layered on top of Ultralytics.
# ``edgefirst_key`` maps to the platform name in the EdgeFirst Studio metrics.
PLATFORMS = {
    "onnx-cpu": {
        "backend": "onnxruntime", "device": "cpu",
        "baseline_validator": "ultralytics", "vendor": None,
        "edgefirst_key": "onnx-cpu",
    },
    "onnx-cuda": {
        "backend": "onnxruntime", "device": "cuda",
        "baseline_validator": "ultralytics", "vendor": None,
        "edgefirst_key": "onnx-cuda",
    },
    "orin-nano-tensorrt": {
        "backend": "tensorrt", "device": "tensorrt",
        "baseline_validator": "ultralytics", "vendor": "nvidia-tensorrt",
        "edgefirst_key": "orin-nano-tensorrt",
    },
    "macos-onnx-coreml": {
        "backend": "onnxruntime-coreml", "device": "coreml",
        "baseline_validator": "ultralytics", "vendor": "apple-coreml",
        "edgefirst_key": "macos-onnx-coreml-ane",
    },
    "rpi5-hailo8l": {
        # Baseline = Hailo Model Zoo PRECOMPILED HEFs (vendor:hailo-model-zoo).
        # Models come from the Model Zoo (reference) or EdgeFirst Studio; we do
        # not compile our own HEFs.
        "backend": "hailort", "device": "hailo",
        "baseline_validator": "yolo-validator", "vendor": "hailo-model-zoo",
        "edgefirst_key": "rpi5-hailo8l",
    },
    "imx95-neutron": {
        # Ultralytics full-integer INT8 TFLite (PTQ), recompiled to Neutron
        # microcode (eIQ neutron-converter) and run via the Neutron delegate.
        "backend": "tflite", "device": "npu",
        "baseline_validator": "yolo-validator", "vendor": "nxp-neutron",
        "edgefirst_key": "imx95-neutron",
    },
    "imx8mp-vsi": {
        # Ultralytics full-integer INT8 TFLite (PTQ), run directly via the
        # VeriSilicon VX delegate on the i.MX 8M Plus NPU (no vendor re-quant).
        "backend": "tflite", "device": "npu",
        "baseline_validator": "yolo-validator", "vendor": None,
        "edgefirst_key": "imx8mp-vsi",
    },
    "imx95-ara240": {
        "backend": "ara", "device": "npu",
        "baseline_validator": "yolo-validator", "vendor": "ara2",
        "edgefirst_key": "imx95-ara240",
    },
}

# Maps a benchmark_a config key to (validator, engine). All baseline configs use
# the Ultralytics export workflow unless a vendor step is recorded per-row.
CONFIG_LANE = {
    "ult-pt":      ("ultralytics", "pytorch"),
    "ult-onnx":    ("ultralytics", "onnx"),
    "ult-engine":  ("ultralytics", "tensorrt"),
    "yv-torch":    ("yolo-validator", "torch"),
    "yv-numpy":    ("yolo-validator", "numpy"),
    "yv-tensorrt": ("yolo-validator", "tensorrt"),
    "yv-hailo":    ("yolo-validator", "hailo"),
    "yv-tflite":   ("yolo-validator", "tflite"),
}


def model_of(variant: str) -> str:
    """Base model name with the export-mode suffix stripped."""
    return variant.replace("-classic", "").replace("-nmsfree", "")
