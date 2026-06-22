#!/usr/bin/env bash
# Fetch official Hailo Model Zoo PRECOMPILED Hailo-8L HEFs for the rpi5-hailo8l
# benchmark. These are the vendor's published artifacts — the "what the vendor
# ships" baseline. Reference HEFs come from the Model Zoo (this script) or from
# EdgeFirst Studio; we do not compile our own.
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

# Vendor precompiled models used as the rpi5-hailo8l baseline. hailo_infer.py
# auto-dispatches by output head: baked-NMS detect (yolov8n/yolov11n/yolov5s),
# NMS-free detect (yolo26n/s/m — raw 4-ch box + cls, host-decoded), and instance
# segmentation (yolov8{n,s,m}_seg — raw DFL box + cls + mask coeffs + proto,
# host-decoded). yolov5*_seg is anchor-based (different head) — not used here.
MODELS=${MODELS:-"yolov8n yolov11n yolov5s yolo26n yolo26s yolo26m yolov8n_seg yolov8s_seg yolov8m_seg"}

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
echo "done. Published Hailo-8L INT8 mAP (cross-check): yolov8n 36.4 | yolov11n 37.8 |"
echo "  yolov5s 34.1 | yolo26n 37.4 | yolo26s 44.8 | yolo26m 50.0 |"
echo "  yolov8n_seg 29.6 mask | yolov8s_seg 36.3 mask | yolov8m_seg 40.2 mask"
