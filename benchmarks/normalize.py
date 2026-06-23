"""Normalize raw benchmark_a results into a committed per-platform metrics file.

Reads ``benchmark_a_*.json`` from a results directory (gitignored) and writes
``benchmarks/metrics/<platform>.json`` (committed) per the schema in
``platforms.py``. One file per platform is the cross-branch merge unit.

Usage::

    python -m benchmarks.normalize \
        --results-dir benchmarks/results/cpu_fp32_nano \
        --platform onnx-cpu --precision FP32

For baseline runs produced by benchmark_a, the converter/quantizer is the
Ultralytics export workflow; pass ``--vendor hailo-model-zoo`` (etc.) and
``--quant-method``/``--calib`` for vendor-quantized artifacts.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from benchmarks.platforms import CONFIG_LANE, PLATFORMS, model_of


def _row(variant, task, precision, quant, validator, engine, vendor, cfg):
    bbox = cfg.get("bbox") or {}
    segm = cfg.get("segm") or {}
    tm = cfg.get("timing") or {}
    return {
        "model": model_of(variant),
        "variant": variant,
        "task": task,
        "precision": precision,
        "quant": quant,
        "lane": "baseline",
        "validator": validator,
        "workflow": f"vendor:{vendor}" if vendor else "ultralytics",
        "engine": engine,
        "box_ap": bbox.get("AP"),
        "box_ap50": bbox.get("AP50"),
        "mask_ap": segm.get("AP"),
        "mask_ap50": segm.get("AP50"),
        "fps_wall": cfg.get("fps_wall"),
        "latency_ms": {
            "pre": tm.get("preprocess"), "inf": tm.get("inference"),
            "post": tm.get("postprocess"), "e2e": tm.get("e2e"),
        },
        "n_images": cfg.get("n_images"),
    }


def normalize(results_dir, platform, precision="FP32",
              quant_method="none", calib=None, vendor=None):
    if platform not in PLATFORMS:
        raise SystemExit(f"unknown platform '{platform}'; "
                         f"known: {', '.join(PLATFORMS)}")
    quant = {"method": quant_method, "calib": calib}
    rows, host = [], None
    files = sorted(glob.glob(os.path.join(results_dir, "benchmark_a_*.json")))
    if not files:
        raise SystemExit(f"no benchmark_a_*.json under {results_dir}")
    for fn in files:
        d = json.load(open(fn, encoding="utf-8"))
        host = host or d.get("host")
        variant, task = d.get("label"), d.get("task")
        for key, cfg in (d.get("configs") or {}).items():
            if not isinstance(cfg, dict) or "error" in cfg or key not in CONFIG_LANE:
                continue
            if cfg.get("bbox") is None:  # config didn't score
                continue
            validator, engine = CONFIG_LANE[key]
            rows.append(_row(variant, task, precision, quant,
                             validator, engine, vendor, cfg))

    os.makedirs("benchmarks/metrics", exist_ok=True)
    path = os.path.join("benchmarks/metrics", f"{platform}.json")

    # Merge by precision: a platform file may hold several precisions (e.g. the
    # onnx-cuda FP32 and FP16 lanes). This run owns one precision, so keep any
    # existing rows of OTHER precisions and replace only this precision's rows.
    # (Within a precision the results-dir must be complete — it fully replaces.)
    kept = []
    if os.path.exists(path):
        prev = json.load(open(path, encoding="utf-8"))
        host = host or prev.get("host")
        kept = [r for r in prev.get("rows", []) if r.get("precision") != precision]
    all_rows = kept + rows

    out = {"platform": platform, "host": host,
           "eval": "crowd-as-normal", "rows": all_rows}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    other = f" (+{len(kept)} rows kept from other precisions)" if kept else ""
    print(f"wrote {path}: {len(all_rows)} rows{other}; this run added "
          f"{len(rows)} {precision} rows "
          f"({len({r['variant'] for r in rows})} variants, "
          f"{len({r['validator'] for r in rows})} validators)")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--platform", required=True, choices=list(PLATFORMS))
    p.add_argument("--precision", default="FP32",
                   choices=["FP32", "FP16", "INT8", "INT16"])
    p.add_argument("--quant-method", default="none")
    p.add_argument("--calib", default=None)
    p.add_argument("--vendor", default=None,
                   help="silicon-vendor workflow id (e.g. hailo-model-zoo, nxp-eiq)")
    a = p.parse_args()
    normalize(a.results_dir, a.platform, a.precision,
              a.quant_method, a.calib, a.vendor)


if __name__ == "__main__":
    main()
