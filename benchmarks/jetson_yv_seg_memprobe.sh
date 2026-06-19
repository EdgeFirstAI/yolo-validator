#!/usr/bin/env bash
# Memory probe: run yolo-validator's TensorRT seg path STANDALONE (no
# Ultralytics, no torch in-process) over full COCO val2017, sampling the
# process's own VmRSS / VmSwap every 3s. Confirms where the seg memory
# pressure lives. Markers: yvseg.DONE / yvseg.FAIL; samples: yvseg.mem.
set -u
cd ~/yolo-validator
ENGINE=benchmarks/results/jetson_full_ult/_models/yolo26n-seg-classic.engine
IMAGES=~/Datasets/COCO/val2017
GT=~/Datasets/COCO/annotations/instances_val2017.json
rm -f yvseg.DONE yvseg.FAIL yvseg.mem yvseg.log

env PYTHONIOENCODING=utf-8 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
  venv-ult/bin/python -m yolo_validator --model "$ENGINE" --runtime tensorrt \
  --images "$IMAGES" --gt "$GT" --task segment --warmup 5 \
  --output yvseg.pred.json > yvseg.log 2>&1 &
PID=$!

peak_rss=0
peak_swap=0
while kill -0 "$PID" 2>/dev/null; do
  rss=$(awk '/^VmRSS/{print $2}' "/proc/$PID/status" 2>/dev/null)
  swp=$(awk '/^VmSwap/{print $2}' "/proc/$PID/status" 2>/dev/null)
  if [ -n "$rss" ]; then
    echo "$(date +%s) rss_kB=$rss swap_kB=${swp:-0}" >> yvseg.mem
    [ "$rss" -gt "$peak_rss" ] && peak_rss=$rss
    [ -n "$swp" ] && [ "$swp" -gt "$peak_swap" ] && peak_swap=$swp
  fi
  sleep 3
done

if wait "$PID"; then touch yvseg.DONE; else touch yvseg.FAIL; fi
echo "PEAK_RSS_kB=$peak_rss PEAK_SWAP_kB=$peak_swap" >> yvseg.mem
echo "PEAK_RSS_MiB=$((peak_rss/1024)) PEAK_SWAP_MiB=$((peak_swap/1024))" >> yvseg.mem
