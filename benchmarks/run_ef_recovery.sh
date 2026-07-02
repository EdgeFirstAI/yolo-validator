#!/bin/bash
# E2E EdgeFirst decoder accuracy-recovery measurement (offline).
# Runs the patched profiler on COCO val2017 (ds-14a1) with the validation
# decode flags OFF (baseline) vs ON (multi_label + per-channel DFL defaults),
# same model+provider, then scores both via the parquet->COCO bridge.
BIN=~/Software/Studio/profiler/target/release/edgefirst-profiler
MODEL=~/Library/Caches/edgefirst-profiler/models/yolov8n-seg-t-27d9.fp16.onnx
DS=~/Library/Caches/edgefirst-profiler/datasets/ds-14a1
OUT=~/Software/Studio/yolo-validator/benchmarks/results/ef_recovery
cd ~/Software/Studio/yolo-validator && source venv/bin/activate
mkdir -p "$OUT"

echo "############ RUN OFF (multi_label=false, dfl_per_channel=false) ############"
time "$BIN" validate --model "$MODEL" --images "$DS" --provider coreml-gpu \
  --multi-label false --dfl-per-channel false --output "$OUT/off" \
  >"$OUT/off.log" 2>&1
echo "off exit=$?  (last log lines:)"; grep -ivE "Device Usage|MLGPUComputeDevice" "$OUT/off.log" | tail -4

echo "############ RUN ON (validation defaults: multi_label + per-channel) ############"
time "$BIN" validate --model "$MODEL" --images "$DS" --provider coreml-gpu \
  --output "$OUT/on" \
  >"$OUT/on.log" 2>&1
echo "on exit=$?  (last log lines:)"; grep -ivE "Device Usage|MLGPUComputeDevice" "$OUT/on.log" | tail -4

echo "############ SCORE OFF (baseline) ############"
python -m benchmarks.score_profiler_parquet --parquet "$OUT/off/predictions.parquet" --iou bbox,segm
echo "############ SCORE ON (validation-accurate) ############"
python -m benchmarks.score_profiler_parquet --parquet "$OUT/on/predictions.parquet" --iou bbox,segm
echo "############ DONE ############"
