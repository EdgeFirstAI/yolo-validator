#!/usr/bin/env bash
# Fetch official Hailo Model Zoo PRECOMPILED Hailo-8L HEFs for the rpi5-hailo8l
# benchmark. These are the vendor's published artifacts — the primary "what the
# vendor ships" baseline. Run our own compile (hailo_compile.py) only to fill
# gaps the Model Zoo does not cover (e.g. yolov5nu, the Ultralytics anchor-free
# v5 retrain, which the zoo does not publish).
#
# Run on the device (or anywhere) where the .hef will be evaluated. Files land
# next to this repo's gitignored results tree.
#
# Provenance / compatibility (verified): Model Zoo release v2.18 HEFs are built
# with Dataflow Compiler v3.33.1 and target HailoRT 4.22/4.23 (the Pi runs
# 4.23). Input is 640x640x3 UINT8 RGB with /255 normalization baked in.
#
# IMPORTANT — eval threshold: the published detection HEFs bake a *deployment*
# NMS score threshold (yolov8n: 0.2) that clips low-confidence boxes and
# depresses COCO mAP ~4-8 pp. hailo_infer.py overrides it to 0.001 at runtime
# (HailoRT InferVStreams.set_nms_score_threshold) for a fair, comparable mAP.
set -u
OUT=${1:-"$(dirname "$0")/results/hailo_zoo"}
BASE=https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l
mkdir -p "$OUT"

# Vendor precompiled detection models used as the rpi5-hailo8l baseline.
# (yolo26n is published but emits raw NMS-free tensors needing a custom host
#  decoder — not wired up here.)
MODELS=${MODELS:-"yolov8n yolov11n yolov5s"}

for m in $MODELS; do
    dst="$OUT/$m.hef"
    if [ -f "$dst" ]; then
        echo "have $m.hef ($(stat -c%s "$dst") bytes)"
        continue
    fi
    echo "fetching $m.hef ..."
    wget -q "$BASE/$m.hef" -O "$dst" || { echo "FAILED: $m"; rm -f "$dst"; exit 1; }
    echo "  -> $dst ($(stat -c%s "$dst") bytes)"
done
echo "done. Published Hailo-8L INT8 mAP (for cross-check): yolov8n 36.4 | yolov11n 37.8 | yolov5s 34.1"
