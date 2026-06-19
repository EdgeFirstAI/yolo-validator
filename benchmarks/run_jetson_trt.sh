#!/usr/bin/env bash
# Jetson yv-tensorrt benchmark: build TRT engines via trtexec (no torch needed)
# and run yolo-validator's TensorRT path over the full COCO val2017 set.
set -u
cd ~/yolo-validator
TRTEXEC=/usr/src/tensorrt/bin/trtexec
IMAGES=~/Datasets/COCO/val2017
GT=~/Datasets/COCO/annotations/instances_val2017.json
OUT=jetson_results
mkdir -p "$OUT"

# model:task pairs (classic + nmsfree variants, detect + segment)
models="yolov8n-classic:detect yolov8n-seg-classic:segment \
yolo26n-classic:detect yolo26n-nmsfree:detect \
yolo26n-seg-classic:segment yolo26n-seg-nmsfree:segment"

for entry in $models; do
    m="${entry%%:*}"; task="${entry##*:}"
    echo "============ $m ($task) ============"
    if [ ! -f "$m.engine" ]; then
        echo "[build] building $m.engine ..."
        if $TRTEXEC --onnx="$m.onnx" --saveEngine="$m.engine" > "$OUT/$m.build.log" 2>&1; then
            echo "[build] ok"
        else
            echo "[build] FAILED (see $OUT/$m.build.log)"; continue
        fi
    fi
    echo "[val] yv-tensorrt over full val2017 ..."
    ./venv/bin/python -m yolo_validator --model "$m.engine" --runtime tensorrt \
        --images "$IMAGES" --gt "$GT" --task "$task" --warmup 5 \
        --output "$OUT/$m.pred.json" > "$OUT/$m.log" 2>&1
    echo "[done] $m:"
    grep -iE "Model-path|bbox:|segm:" "$OUT/$m.log" | sed 's/^/    /'
    touch "$OUT/$m.DONE"
done
echo ALL_DONE
