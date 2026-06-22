"""Compare EdgeFirst Studio (Profiler + Validator) results against the
Ultralytics baseline and the yolo-validator proxy, on the platforms this repo
benchmarks.

Two axes:
  * **Accuracy** (mAP, percentage points) — the directly comparable axis.
  * **Performance** (throughput + per-stage latency) — EdgeFirst's pipelining
    is a core benefit, so we explicitly flag any config where EdgeFirst is NOT
    faster than the single-stream reference on MATCHED hardware. Those are
    publish-blockers to analyze before release.

Data sources
------------
- **Ours**: normalized rows in ``benchmarks/metrics/<platform>.json``
  (``validator`` = ``ultralytics`` baseline or ``yolo-validator`` proxy;
  ``engine`` = onnx/pytorch/numpy/torch/tensorrt/hailo).
- **EdgeFirst**: fetched live per validation session via
  :mod:`benchmarks.studio_fetch`. The ``--catalog`` (EdgeFirst Studio metrics
  export) is used ONLY as the session catalog (join keys + ``session_id``).

Performance — IMPORTANT CAVEAT
------------------------------
This tool fetches Studio **VALIDATION** sessions, which are the right source for
ACCURACY but NOT for throughput. The validation run is not the
edgefirst-profiler benchmark: older detection sessions (profiler 1.3.2) record
no ``realized_fps`` at all, and where it exists it is the validation run's
throughput, not the profiler's pipelined bench. We therefore only display
validation-session fps where actually recorded (never fabricated from 1000/e2e)
and do NOT compute speedups/concerns from it. Authoritative EdgeFirst throughput
comes from the on-device profiler bench (``hailortcli`` / ``bench-internal``);
feed those numbers into BENCHMARK.md directly.

Usage::

    python -m benchmarks.compare_edgefirst --catalog ~/.../edgefirst-model-zoo-metrics.json
    python -m benchmarks.compare_edgefirst --catalog ... --refresh   # re-fetch live
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from benchmarks.studio_fetch import edgefirst_canonical

METRICS_DIR = Path(__file__).resolve().parent / "metrics"
CACHE_PATH = Path(__file__).resolve().parent / "results" / "edgefirst_studio_cache.json"
OUR_PLATFORMS = {"onnx-cpu", "onnx-cuda", "orin-nano-tensorrt", "rpi5-hailo8l"}

# Platforms where EdgeFirst and our runs share the device AND runtime family, so
# a throughput comparison is valid. onnx-cpu is excluded (Graviton vs i9).
HW_MATCHED = {"onnx-cuda", "orin-nano-tensorrt", "rpi5-hailo8l"}

# Preferred engine per lane: ultralytics = the canonical validator; yolo-validator
# = the portable single-stream path that matches EdgeFirst's runtime.
ULT_ENGINES = ("pytorch", "tensorrt", "onnx")
YV_ENGINES = ("numpy", "tensorrt", "hailo", "torch")


def _rows_of(doc):
    if isinstance(doc, list):
        return doc
    for key in ("records", "rows", "metrics", "data"):
        if isinstance(doc.get(key), list):
            return doc[key]
    return [doc]


def load_ours() -> dict:
    """(platform, model, task, precision) -> list[row] (all engines/lanes)."""
    index: dict = {}
    for path in glob.glob(str(METRICS_DIR / "*.json")):
        platform = os.path.basename(path).replace(".json", "")
        for row in _rows_of(json.load(open(path))):
            key = (platform, row["model"], row["task"], row["precision"])
            index.setdefault(key, []).append(row)
    return index


def pick(rows, validator, engine_prefs):
    """Choose one row for a lane, preferring engines in order."""
    cands = [r for r in rows if r.get("validator") == validator]
    for eng in engine_prefs:
        for r in cands:
            if r.get("engine") == eng:
                return r
    return cands[0] if cands else None


def _candidates(rec) -> list[str]:
    """Our model name(s) an EdgeFirst catalog row could map to.

    EdgeFirst's ``yolov5-det`` is the anchor-free retrain, so it maps ONLY to
    our ``yolov5*u`` names — never the classic anchor-based ``yolov5s`` (rpi5
    vendor HEF), which is a different model.
    """
    version, size = rec["version"], rec["size"]
    suffix = "-seg" if rec["task"] == "seg" else ""
    if version == "yolov5":
        return [f"yolov5{size}u{suffix}"]
    return [f"{version}{size}{suffix}"]


def match_catalog(catalog_path: Path, ours: dict) -> list[dict]:
    doc = json.load(open(catalog_path))
    matched, seen, dropped = [], set(), 0
    for rec in doc["records"]:
        if rec["platform"] not in OUR_PLATFORMS:
            continue
        task = "segment" if rec["task"] == "seg" else "detect"
        for cand in _candidates(rec):
            key = (rec["platform"], cand, task, rec["precision"])
            if key not in ours:
                continue
            if key in seen:
                dropped += 1
                break
            seen.add(key)
            matched.append({"key": key, "session_id": rec["session_id"], "ef": rec})
            break
    if dropped:
        print(f"note: dropped {dropped} duplicate EdgeFirst session(s) mapping to an "
              f"already-matched lane (kept first per key)")
    return matched


def _load_cache() -> dict:
    return json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}


def fetch_edgefirst(session_ids, refresh) -> dict:
    cache = {} if refresh else _load_cache()
    todo = [s for s in session_ids if s not in cache]
    for i, sid in enumerate(todo, 1):
        print(f"  fetching {sid}  ({i}/{len(todo)})", flush=True)
        try:
            cache[sid] = edgefirst_canonical(sid)
        except Exception as exc:
            print(f"    ! {sid}: {exc}")
            cache[sid] = {"error": str(exc)}
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1))
    return cache


def _pp(frac):
    return None if frac is None else 100.0 * frac


def _f(val, w=6, p=2):
    return f"{'—':>{w}}" if val is None else f"{val:{w}.{p}f}"


def accuracy_report(matched, ours, ef):
    print("\n========== ACCURACY (mAP, percentage points) ==========")
    gaps = []
    for m in sorted(matched, key=lambda x: (x["key"][0], x["key"][2], x["key"][1])):
        plat, model, task, prec = m["key"]
        rows = ours[m["key"]]
        ult, yv = pick(rows, "ultralytics", ULT_ENGINES), pick(rows, "yolo-validator", YV_ENGINES)
        efc = ef.get(m["session_id"], {})
        if task == "detect":
            ef_v = _pp((efc.get("bbox") or {}).get("AP"))
            ult_v, yv_v = _pp((ult or {}).get("box_ap")), _pp((yv or {}).get("box_ap"))
        else:
            ef_v = _pp((efc.get("segm") or {}).get("AP"))
            ult_v, yv_v = _pp((ult or {}).get("mask_ap")), _pp((yv or {}).get("mask_ap"))
        base_v, lane = (ult_v, "ult") if ult_v is not None else (yv_v, "yv")
        if ef_v is not None and base_v is not None:
            gaps.append((plat, model, task, prec, lane, ef_v, base_v, ef_v - base_v))
    print("primary metric: detection = box mAP@0.5:0.95 ; segmentation = mask mAP@0.5:0.95")
    for g in gaps:
        plat, model, task, prec, lane, ef_v, base_v, d = g
        print(f"  {plat:18s} {model:14s} {task:7s} {prec:5s} | "
              f"EF {ef_v:6.2f} vs {lane:3s} {base_v:6.2f} | Δ {d:+5.2f} pp")
    if gaps:
        ds = [g[7] for g in gaps]
        print(f"  n={len(ds)}  mean Δ {sum(ds)/len(ds):+.2f} pp  "
              f"min {min(ds):+.2f}  max {max(ds):+.2f}")


def perf_report(matched, ours, ef):
    print("\n========== PERFORMANCE (validation-session throughput — NOT authoritative) ==========")
    print("⚠ DO NOT draw throughput conclusions from this table. The EF numbers come from")
    print("  Studio VALIDATION sessions, which are not the edgefirst-profiler benchmark:")
    print("  - older detection sessions (profiler 1.3.2) record NO realized_fps -> shown 'n/a'")
    print("    (we no longer fabricate 1000/e2e, which previously faked a serial throughput).")
    print("  - where present (newer seg sessions) it is the validation run's throughput, still")
    print("    not the profiler's pipelined bench.")
    print("  Authoritative EdgeFirst throughput = on-device profiler bench (hailortcli /")
    print("  bench-internal). yv fps_wall is our single-stream reference.\n")
    for plat in sorted({m["key"][0] for m in matched}):
        print(f"### {plat}")
        print("model          task    prec  |  EF fps(val)  yv fps  |  EF e2e  yv e2e ms")
        print("-" * 74)
        for m in sorted([x for x in matched if x["key"][0] == plat], key=lambda x: x["key"][1]):
            _, model, task, prec = m["key"]
            yv = pick(ours[m["key"]], "yolo-validator", YV_ENGINES)
            efc = ef.get(m["session_id"], {})
            ef_fps = efc.get("fps_pipeline")  # None when the session didn't record it
            eft = efc.get("timing") or {}
            yv_lat = (yv or {}).get("latency_ms") or {}
            ef_fps_s = "   n/a" if ef_fps is None else f"{ef_fps:6.1f}"
            print(f"{model:14s} {task:7s} {prec:5s} |   {ef_fps_s}     {_f((yv or {}).get('fps_wall'))}  |"
                  f"  {_f(eft.get('e2e'))} {_f(yv_lat.get('e2e'))}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True, type=Path,
                    help="EdgeFirst Studio metrics export (session catalog)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch every session via studio_fetch (ignore cache)")
    args = ap.parse_args()

    ours = load_ours()
    matched = match_catalog(args.catalog.expanduser(), ours)
    print(f"matched {len(matched)} EdgeFirst sessions to our metrics; "
          f"fetching via studio_fetch (cache: {CACHE_PATH.name})")
    ef = fetch_edgefirst([m["session_id"] for m in matched], args.refresh)
    accuracy_report(matched, ours, ef)
    perf_report(matched, ours, ef)


if __name__ == "__main__":
    main()
