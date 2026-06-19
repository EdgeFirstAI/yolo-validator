#!/usr/bin/env bash
# Official Ultralytics Jetson install (JetPack 6 / cp310 aarch64), per
# https://docs.ultralytics.com/guides/nvidia-jetson
# Sets up a venv with Ultralytics + its Jetson torch/torchvision/onnxruntime-gpu
# wheels (github.com/ultralytics/assets) + cuDSS, then yolo-validator, so the
# DIRECT on-device Ultralytics comparison (ult-pt cuda / ult-engine) can run.
# Detached-friendly: logs to install.log, touches install.DONE / install.FAIL.
set -u
cd ~/yolo-validator
LOG=install.log
: > "$LOG"
rm -f install.DONE install.FAIL
exec >>"$LOG" 2>&1

fail() { echo "FAILED at: $1"; touch install.FAIL; exit 1; }

A=https://github.com/ultralytics/assets/releases/download/v0.0.0
TORCH_WHL=$A/torch-2.10.0-cp310-cp310-linux_aarch64.whl
TV_WHL=$A/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl
ORT_WHL=$A/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
CUDSS_DEB=cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
CUDSS_URL=https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/$CUDSS_DEB

echo "=== [1/7] fresh venv (--system-site-packages for system TensorRT 10.3 + pycuda) ==="
rm -rf venv-ult
python3 -m venv --system-site-packages venv-ult || fail "venv create"
PY=venv-ult/bin/python
$PY -m pip install -U pip || fail "pip upgrade"

echo "=== [2/7] ultralytics[export] ==="
$PY -m pip install 'ultralytics[export]' || fail "ultralytics"

echo "=== [3/7] overwrite torch/torchvision with Ultralytics Jetson wheels ==="
$PY -m pip install "$TORCH_WHL" "$TV_WHL" || fail "jetson torch wheels"

echo "=== [4/7] cuDSS (torch 2.10 dep) via apt ==="
if $PY -c "import torch" 2>/dev/null; then
    echo "torch imports already; checking if cuDSS needed..."
fi
if ! $PY -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    echo "installing cuDSS..."
fi
wget -q "$CUDSS_URL" -O "$CUDSS_DEB" || fail "cudss download"
sudo dpkg -i "$CUDSS_DEB" || fail "cudss dpkg"
sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/ || fail "cudss keyring"
sudo apt-get update || fail "apt update"
sudo apt-get -y install cudss || fail "apt cudss"

echo "=== [5/7] onnxruntime-gpu Jetson wheel ==="
$PY -m pip install "$ORT_WHL" || fail "onnxruntime-gpu"

echo "=== [6/7] yolo-validator (+ benchmark deps) into same env ==="
# --no-deps so we keep the Jetson torch/onnxruntime; add only what yv needs.
$PY -m pip install pycocotools || fail "pycocotools"
# pycuda — required by yolo-validator's TensorRT backend (yv-tensorrt). The
# JetPack image does not ship it in system site-packages, so build it from
# source against the CUDA toolkit (nvcc must be on PATH; no aarch64 wheel).
export PATH=/usr/local/cuda/bin:$PATH
$PY -m pip install pycuda || fail "pycuda"
$PY -m pip install --no-deps -e . || fail "yolo-validator editable"

echo "=== [7/7] SMOKE TEST ==="
$PY - <<'PYEOF' || fail "smoke test"
import torch, sys
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
import torchvision; print("torchvision:", torchvision.__version__)
import ultralytics; print("ultralytics:", ultralytics.__version__)
import tensorrt; print("tensorrt:", tensorrt.__version__)
import pycuda.autoinit, pycuda.driver as drv
print("pycuda:", drv.get_version())
try:
    import onnxruntime as ort
    print("onnxruntime:", ort.__version__, "providers:", ort.get_available_providers())
except Exception as e:
    print("onnxruntime import warn:", e)
import yolo_validator; print("yolo_validator: OK")
assert torch.cuda.is_available(), "torch.cuda NOT available — Jetson GPU path would fall back to CPU"
print("SMOKE OK")
PYEOF

echo "ALL_DONE"
touch install.DONE
