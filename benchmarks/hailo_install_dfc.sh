#!/usr/bin/env bash
# Hailo Dataflow Compiler (DFC) environment for the rpi5-hailo8l benchmark.
#
# The DFC is the **compile** half of the Ultralytics/Vendor Hailo workflow
# (https://docs.ultralytics.com/integrations/hailo): on an x86_64 Linux host it
# turns an Ultralytics-exported ONNX into a quantized INT8 .hef. On-device
# inference (HailoRT on the Raspberry Pi 5 AI Kit) is a separate environment.
#
# Why this is not a plain `pip install` into ./venv:
#   * DFC 3.33.1 hard-pins numpy==1.26.4 / onnx==1.16.0 / protobuf==3.20.3 and
#     pulls TensorFlow 2.18 — installing into ./venv would downgrade and break
#     the benchmark stack. So we build a dedicated, isolated `venv-hailo`.
#   * The DFC is validated on Python 3.8-3.10; on 3.12 its onnxsim==0.4.36 pin
#     has no wheel and fails to compile. We use a standalone CPython 3.10 via
#     `uv` (user-local, no sudo, nothing touches /usr).
#   * pygraphviz (a hard dep) needs the system graphviz C headers. It is used
#     ONLY for graph visualization; the SDK already supports a headless path
#     (has_graphviz=False). We install everything EXCEPT pygraphviz and drop a
#     tiny stub so `import pygraphviz` succeeds and the SDK degrades gracefully
#     — no graphviz, no sudo. Parse/optimize/quantize/compile are unaffected.
#
# Idempotent and detached-friendly: logs to venv-hailo-install.log, touches
# venv-hailo-install.DONE / .FAIL.
set -u
cd "$(dirname "$0")/.." || exit 1   # repo root
LOG=venv-hailo-install.log
: > "$LOG"
rm -f venv-hailo-install.DONE venv-hailo-install.FAIL
exec >>"$LOG" 2>&1

fail() { echo "FAILED at: $1"; touch venv-hailo-install.FAIL; exit 1; }

WHL=${HAILO_DFC_WHL:-$(ls -1 "$HOME"/hailo/hailo_dataflow_compiler-*-py3-none-linux_x86_64.whl 2>/dev/null | head -1)}
[ -n "$WHL" ] && [ -f "$WHL" ] || fail "DFC wheel not found (set HAILO_DFC_WHL or place it in ~/hailo/)"
echo "=== DFC wheel: $WHL ==="

echo "=== [1/5] ensure uv (user-local, no sudo) ==="
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    # Download then run as separate steps so a failed download is not masked by
    # the pipe's exit status (and the installer is auditable before it runs).
    curl -LsSf https://astral.sh/uv/install.sh -o uv-install.sh || fail "uv download"
    sh uv-install.sh || fail "uv install"
    rm -f uv-install.sh
fi
uv --version || fail "uv not on PATH"

echo "=== [2/5] standalone CPython 3.10 + fresh venv-hailo ==="
uv python install 3.10 || fail "uv python 3.10"
rm -rf venv-hailo
uv venv --python 3.10 venv-hailo || fail "uv venv"

echo "=== [3/5] install DFC deps EXCEPT pygraphviz ==="
# pygraphviz needs the system graphviz C library; we stub it (see [4/5]). Pull
# the DFC's Requires-Dist straight from the wheel so the pin set stays exact.
venv-hailo/bin/python - "$WHL" > hailo_dfc_reqs.txt <<'PY' || fail "extract reqs"
import sys, zipfile, re
z = zipfile.ZipFile(sys.argv[1])
meta = next(n for n in z.namelist() if n.endswith('METADATA'))
for line in z.read(meta).decode().splitlines():
    if line.startswith('Requires-Dist:'):
        dep = line.split(':', 1)[1].strip()
        if re.match(r'pygraphviz', dep, re.I):
            continue
        print(dep)
PY
uv pip install --python venv-hailo/bin/python -r hailo_dfc_reqs.txt || fail "DFC deps"
uv pip install --python venv-hailo/bin/python --no-deps "$WHL" || fail "DFC wheel"
rm -f hailo_dfc_reqs.txt

echo "=== [4/5] headless pygraphviz stub ==="
SP=$(venv-hailo/bin/python -c "import site; print(site.getsitepackages()[0])") || fail "site-packages"
cat > "$SP/pygraphviz.py" <<'PY'
"""Headless stub for pygraphviz (yolo-validator Hailo bench environment).

The Hailo Dataflow Compiler imports pygraphviz only to render model-graph
visualizations. Its __init__ probes graphviz via ``AGraph().layout("dot")`` and,
on ``(ValueError, OSError)``, sets ``has_graphviz = False`` and skips all
visualization — but it does NOT catch ``ModuleNotFoundError``, so a fully absent
pygraphviz crashes import instead of degrading. This stub makes ``import
pygraphviz`` succeed while ``layout()`` raises OSError, driving the SDK down its
own supported headless path with no system graphviz C library (no sudo).
"""


class AGraph:
    def __init__(self, *args, **kwargs):
        pass

    def layout(self, *args, **kwargs):
        raise OSError("pygraphviz stub: graphviz unavailable (headless bench env)")

    def __getattr__(self, name):
        raise OSError(f"pygraphviz stub: '{name}' unavailable (headless bench env)")
PY

echo "=== [5/5] SMOKE TEST ==="
venv-hailo/bin/python - <<'PY' || fail "smoke test"
import hailo_sdk_client
from hailo_sdk_client import ClientRunner
ClientRunner(hw_arch="hailo8l")
import numpy as np, onnx, onnxruntime
print("hailo_sdk_client + ClientRunner(hailo8l): OK")
print("numpy:", np.__version__, "onnx:", onnx.__version__, "ort:", onnxruntime.__version__)
print("SMOKE OK")
PY

echo "ALL_DONE"
touch venv-hailo-install.DONE
