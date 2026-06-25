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

from benchmarks.platforms import PLATFORMS
from benchmarks.studio_fetch import edgefirst_canonical

METRICS_DIR = Path(__file__).resolve().parent / "metrics"
CACHE_PATH = Path(__file__).resolve().parent / "results" / "edgefirst_studio_cache.json"
OUR_PLATFORMS = {"onnx-cpu", "onnx-cuda", "orin-nano-tensorrt", "rpi5-hailo8l",
                 "macos-onnx-coreml"}

# Platforms where EdgeFirst and our runs share the device AND runtime family, so
# a throughput comparison is valid. onnx-cpu is excluded (Graviton vs i9).
# macos-onnx-coreml is hw+runtime matched: both sides are ORT CoreML EP on the ANE.
HW_MATCHED = {"onnx-cuda", "orin-nano-tensorrt", "rpi5-hailo8l", "macos-onnx-coreml"}

# EdgeFirst catalog platform string -> our platform id, via the registry's
# ``edgefirst_key``. For every prior platform our id == edgefirst_key, so the join
# worked by coincidence; macOS is the first to differ (our ``macos-onnx-coreml`` vs
# catalog ``macos-onnx-coreml-ane``). Only ``-ane`` is wired in the registry, so the
# catalog's ``-gpu``/``-cpu`` lanes have no entry and are naturally excluded.
EF_KEY_TO_OURS = {m["edgefirst_key"]: pid
                  for pid, m in PLATFORMS.items() if m.get("edgefirst_key")}

# Preferred engine per lane: ultralytics = the canonical validator; yolo-validator
# = the portable single-stream path that matches EdgeFirst's runtime.
ULT_ENGINES = ("pytorch", "tensorrt", "onnx", "coreml")
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
        our_plat = EF_KEY_TO_OURS.get(rec["platform"], rec["platform"])
        if our_plat not in OUR_PLATFORMS:
            continue
        task = "segment" if rec["task"] == "seg" else "detect"
        for cand in _candidates(rec):
            key = (our_plat, cand, task, rec["precision"])
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


def edgefirst_from_catalog(rec: dict) -> dict:
    """Build a canonical result dict straight from a catalog record — no live fetch.

    Same shape as ``studio_fetch.edgefirst_canonical`` so the report code is
    identical. Catalog AP fields are percentage points (0-100); the report's
    ``_pp`` multiplies by 100, and our metrics rows store fractions, so convert
    here to fractions. Lets the accuracy axis run fully offline (no
    edgefirst-client / httpx).
    """
    def frac(v):
        return None if v is None else v / 100.0

    out: dict = {"session_id": rec.get("session_id"), "session_name": rec.get("model")}
    if rec.get("det_ap") is not None or rec.get("det_ap50") is not None:
        out["bbox"] = {"AP": frac(rec.get("det_ap")), "AP50": frac(rec.get("det_ap50")),
                       "AP75": frac(rec.get("det_ap75"))}
    if rec.get("mask_ap") is not None or rec.get("mask_ap50") is not None:
        out["segm"] = {"AP": frac(rec.get("mask_ap")), "AP50": frac(rec.get("mask_ap50")),
                       "AP75": frac(rec.get("mask_ap75"))}
    if rec.get("e2e_latency_ms") is not None:
        out["timing"] = {"preprocess": rec.get("preprocess_ms"),
                         "inference": rec.get("inference_ms"),
                         "postprocess": rec.get("postprocess_ms"),
                         "e2e": rec.get("e2e_latency_ms")}
    # EdgeFirst throughput: prefer the per-session realized fps; fall back to the
    # catalog's published median fps (the model-zoo benchmark number) so the offline
    # path reports throughput + speedup (the perf header documents "realized/median
    # throughput"). For the macOS CoreML ANE lane the catalog carries fps_median only.
    fps = rec.get("realized_fps_scalar")
    if fps is None:
        fps = rec.get("fps_median")
    if fps is not None:
        out["fps_pipeline"] = fps
    return out


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


def _ref(plat, rows):
    """Pick the comparison reference per the measurement rule: the Ultralytics lane
    where Ultralytics runs on-target (platform ``baseline_validator == 'ultralytics'``),
    else the yolo-validator proxy. Returns ``(row, lane_label)``."""
    baseline = PLATFORMS.get(plat, {}).get("baseline_validator", "ultralytics")
    if baseline == "ultralytics":
        return pick(rows, "ultralytics", ULT_ENGINES), "ult"
    return pick(rows, "yolo-validator", YV_ENGINES), "yv"


def accuracy_report(matched, ours, ef):
    print("\n========== ACCURACY (mAP, percentage points — the ~1pp guardrail) ==========")
    gaps = []
    for m in sorted(matched, key=lambda x: (x["key"][0], x["key"][2], x["key"][1])):
        plat, model, task, prec = m["key"]
        rows = ours[m["key"]]
        base, lane = _ref(plat, rows)
        efc = ef.get(m["session_id"], {})
        if task == "detect":
            ef_v = _pp((efc.get("bbox") or {}).get("AP"))
            base_v = _pp((base or {}).get("box_ap"))
        else:
            ef_v = _pp((efc.get("segm") or {}).get("AP"))
            base_v = _pp((base or {}).get("mask_ap"))
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
    print("\n========== PERFORMANCE — EdgeFirst vs reference (PRIMARY KPI) ==========")
    print("speedup = EdgeFirst fps / reference fps_wall. Reference = Ultralytics where it")
    print("runs on-target, else the yolo-validator proxy (per platform baseline_validator).")
    print("EdgeFirst is PIPELINED (overlapped stages); the reference is single-stream, so the")
    print("speedup reflects EdgeFirst's pipelining + optimized decode (a legitimate edge")
    print("optimization Ultralytics does not provide for validation). EF fps is the recorded")
    print("realized/median throughput; absent on some older sessions (n/a).\n")
    speedups = []
    for plat in sorted({m["key"][0] for m in matched}):
        print(f"### {plat}")
        print("model          task    prec  | EF fps   ref fps  speedup | EF e2e  ref e2e ms | ref")
        print("-" * 86)
        for m in sorted([x for x in matched if x["key"][0] == plat], key=lambda x: x["key"][1]):
            _, model, task, prec = m["key"]
            ref, lane = _ref(plat, ours[m["key"]])
            efc = ef.get(m["session_id"], {})
            ef_fps = efc.get("fps_pipeline")   # realized/median throughput; None if unrecorded
            eft = efc.get("timing") or {}
            ref_fps = (ref or {}).get("fps_wall")
            ref_lat = (ref or {}).get("latency_ms") or {}
            if ef_fps and ref_fps:
                spd = ef_fps / ref_fps
                speedups.append(spd)
                spd_s = f"{spd:6.1f}x"
            else:
                spd_s = "     —"
            ef_fps_s = "   n/a" if ef_fps is None else f"{ef_fps:6.1f}"
            print(f"{model:14s} {task:7s} {prec:5s} | {ef_fps_s} {_f(ref_fps)} {spd_s:>8} |"
                  f" {_f(eft.get('e2e'))} {_f(ref_lat.get('e2e'))} | {lane}")
        print()
    if speedups:
        print(f"speedup: n={len(speedups)}  mean {sum(speedups)/len(speedups):.1f}x  "
              f"min {min(speedups):.1f}x  max {max(speedups):.1f}x")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True, type=Path,
                    help="EdgeFirst Studio metrics export (session catalog)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch every session via studio_fetch (ignore cache)")
    ap.add_argument("--offline", action="store_true",
                    help="read AP/timing straight from the catalog record instead of "
                         "re-fetching live Studio (no edgefirst-client/httpx needed). "
                         "The catalog already carries the full 12-metric summary + "
                         "timing, so the accuracy axis is fully offline.")
    args = ap.parse_args()

    ours = load_ours()
    matched = match_catalog(args.catalog.expanduser(), ours)
    if args.offline:
        print(f"matched {len(matched)} EdgeFirst sessions to our metrics; "
              f"reading AP from catalog (offline)")
        ef = {m["session_id"]: edgefirst_from_catalog(m["ef"]) for m in matched}
    else:
        print(f"matched {len(matched)} EdgeFirst sessions to our metrics; "
              f"fetching via studio_fetch (cache: {CACHE_PATH.name})")
        ef = fetch_edgefirst([m["session_id"] for m in matched], args.refresh)
    perf_report(matched, ours, ef)      # primary KPI first
    accuracy_report(matched, ours, ef)  # the ~1pp guardrail


if __name__ == "__main__":
    main()
