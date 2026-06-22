# benchmarks/studio_fetch.py
"""Fetch EdgeFirst Studio validation metrics and map to the canonical benchmark schema.

Calls the Studio JSON-RPC API directly using the JWT cached by
``edgefirst-client``. Requires no imports from outside this repository —
only stdlib + httpx (install separately if not present: pip install httpx).

CLI usage (debugging / manual verification):
    python -m benchmarks.studio_fetch v-1a8f
    python -m benchmarks.studio_fetch v-1a8f --raw
"""
from __future__ import annotations

import base64
import json
import subprocess
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Auth + transport
# ---------------------------------------------------------------------------

def _get_token() -> str:
    """Return the JWT cached by edgefirst-client login."""
    return subprocess.check_output(["edgefirst-client", "token"], text=True).strip()


def _decode_jwt_payload(token: str) -> dict:
    """Decode the JWT payload without verifying the signature."""
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    pad = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(pad))
    except Exception:
        return {}


def _api_url(token: str) -> str:
    """Derive the Studio JSON-RPC endpoint URL from the JWT payload."""
    claims = _decode_jwt_payload(token)
    url = claims.get("url")
    if isinstance(url, str) and url:
        return url
    server = (claims.get("server") or claims.get("database") or "test").strip()
    return f"https://{server}.edgefirst.studio/api"


def _rpc(method: str, params: dict, token: str, api_url: str) -> dict:
    """Make a single JSON-RPC 2.0 POST to the Studio API."""
    import httpx
    resp = httpx.post(
        api_url,
        headers={"Authorization": f"Bearer {token}"},
        json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(
            f"Studio RPC error: {data['error'].get('message', data['error'])}"
        )
    return data["result"]


def _decode_session_id(hex_id: str) -> int:
    """Decode a display session ID (e.g. 'v-1a8f') to its integer primary key."""
    _, hex_part = hex_id.split("-", 1)
    return int(hex_part, 16)


# ---------------------------------------------------------------------------
# Parsed metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class StudioMetrics:
    """Parsed metrics for one Studio validation session.

    Schema source: ``validate.session.metrics`` JSON-RPC response.

    Detection and segmentation AP come from the Studio COCO evaluator — the
    same eval protocol as the COCO API, producing standard mAP@[0.5:0.95],
    mAP@0.5, and mAP@0.75.

    Timing fields come from ``timing.inline.stages`` — per-frame latencies
    measured on-target while the pipeline was running (stages may overlap due
    to pipelining). ``latency_mean_ms`` is the true end-to-end per-frame
    latency; ``realized_fps`` is the actual pipeline throughput.
    """
    session_id: str
    session_name: str

    # Detection — COCO mAP evaluation
    det_map_50_95: float | None = None
    det_map_50: float | None = None
    det_map_75: float | None = None

    # Segmentation — COCO mAP evaluation (present only for seg model sessions)
    seg_map_50_95: float | None = None
    seg_map_50: float | None = None
    seg_map_75: float | None = None

    # Per-frame stage timing (ms) — from timing.inline.stages
    avg_input_ms: float | None = None       # capture + preprocess (image IO + letterbox + normalize)
    avg_inference_ms: float | None = None   # model inference
    avg_output_ms: float | None = None      # postprocess (NMS + decode + mask materialize)

    # Pipeline-level timing
    latency_mean_ms: float | None = None    # end-to-end per-frame latency (pipelined run)
    realized_fps: float | None = None       # actual throughput (pipeline, not 1000/latency)


# ---------------------------------------------------------------------------
# Fetch functions
# ---------------------------------------------------------------------------

def get_raw_metrics(session_id: str) -> dict:
    """Return the raw ``validate.session.metrics`` JSON-RPC response.

    Useful for inspecting what Studio actually returns — call with ``--raw``
    from the CLI.
    """
    token = _get_token()
    return _rpc(
        "validate.session.metrics",
        {"validate_session_id": _decode_session_id(session_id)},
        token,
        _api_url(token),
    )


def fetch_studio_metrics(session_id: str) -> StudioMetrics:
    """Fetch and parse metrics for one Studio validation session."""
    token = _get_token()
    api_url = _api_url(token)
    int_id = _decode_session_id(session_id)

    session = _rpc("validate.session.get", {"validate_session_id": int_id}, token, api_url)
    name = session.get("docker_task", {}).get("name", session_id)

    raw = _rpc("validate.session.metrics", {"validate_session_id": int_id}, token, api_url)
    if not raw:
        return StudioMetrics(session_id=session_id, session_name=name)

    # Detection COCO mAP — raw["detection"]["summary"]
    det_summary = raw.get("detection", {}).get("summary", {})

    # Segmentation COCO mAP — raw["segmentation"]["summary"] (seg model sessions only)
    seg_summary = raw.get("segmentation", {}).get("summary", {})

    # Timing — raw["timing"]["inline"]["stages"], per-frame measurements
    inline = raw.get("timing", {}).get("inline", {})
    stages = inline.get("stages", {})

    capture_ms = (stages.get("capture_ms") or {}).get("mean_ms")
    preprocess_ms = (stages.get("preprocess_ms") or {}).get("mean_ms")
    inference_ms = (stages.get("inference_ms") or {}).get("mean_ms")
    postprocess_ms = (stages.get("postprocess_ms") or {}).get("mean_ms")

    # Combine capture + preprocess into a single "input" stage — mirrors how
    # yv-torch and ult-onnx report "preprocess" (image load + letterbox + normalize).
    avg_input_ms: float | None = None
    if capture_ms is not None and preprocess_ms is not None:
        avg_input_ms = capture_ms + preprocess_ms
    elif preprocess_ms is not None:
        avg_input_ms = preprocess_ms
    elif capture_ms is not None:
        avg_input_ms = capture_ms

    latency_mean_ms = inline.get("latency_summary", {}).get("mean")
    realized_fps = inline.get("realized_fps_scalar")

    return StudioMetrics(
        session_id=session_id,
        session_name=name,
        det_map_50_95=det_summary.get("AP") or None,
        det_map_50=det_summary.get("AP50") or None,
        det_map_75=det_summary.get("AP75") or None,
        seg_map_50_95=seg_summary.get("AP") or None,
        seg_map_50=seg_summary.get("AP50") or None,
        seg_map_75=seg_summary.get("AP75") or None,
        avg_input_ms=avg_input_ms,
        avg_inference_ms=inference_ms,
        avg_output_ms=postprocess_ms,
        latency_mean_ms=latency_mean_ms,
        realized_fps=realized_fps,
    )


def edgefirst_canonical(session_id: str) -> dict:
    """Fetch a Studio session and return a canonical benchmark result dict.

    The returned dict follows the same schema as ``run_ultralytics()`` /
    ``run_yolo_validator()`` so it slots directly into
    ``results_per_config["edgefirst-profiler"]`` in benchmark_a.py.

    Accuracy mapping:
        bbox.AP    = det_map_50_95  (COCO mAP@[0.5:0.95])
        bbox.AP50  = det_map_50
        bbox.AP75  = det_map_75
        segm.AP    = seg_map_50_95  (COCO mAP@[0.5:0.95], seg models only)
        segm.AP50  = seg_map_50
        segm.AP75  = seg_map_75

    Timing mapping (per-frame mean ms, measured on-target with pipeline running):
        preprocess  = capture_ms + preprocess_ms  (image decode + letterbox + normalize)
        inference   = inference_ms  (ONNX model compute)
        postprocess = postprocess_ms  (NMS + mask materialize + encode)
        e2e         = latency_summary.mean  (actual per-frame latency)

    FPS:
        fps_pipeline = realized_fps_scalar  (actual pipelined throughput — uses
                       parallelism, not comparable 1:1 to single-threaded fps_wall).
        fps_wall is absent — the Studio job total clock includes cloud batch
        queue time and annotation fetch, making it incomparable to local wall.
    """
    m = fetch_studio_metrics(session_id)

    result: dict = {
        "session_id": session_id,
        "session_name": m.session_name,
    }

    if m.det_map_50_95 is not None or m.det_map_50 is not None:
        result["bbox"] = {
            "AP": m.det_map_50_95,
            "AP50": m.det_map_50,
            "AP75": m.det_map_75,
        }

    if m.seg_map_50_95 is not None or m.seg_map_50 is not None:
        result["segm"] = {
            "AP": m.seg_map_50_95,
            "AP50": m.seg_map_50,
            "AP75": m.seg_map_75,
        }

    if m.avg_input_ms is not None:
        # e2e: prefer the measured latency_mean_ms; fall back to sum of stages.
        e2e = m.latency_mean_ms or (
            (m.avg_input_ms or 0.0)
            + (m.avg_inference_ms or 0.0)
            + (m.avg_output_ms or 0.0)
        )
        result["timing"] = {
            "preprocess": m.avg_input_ms,
            "inference": m.avg_inference_ms,
            "postprocess": m.avg_output_ms,
            "e2e": e2e,
        }
        # Store the validation run's MEASURED throughput only. Do NOT fabricate
        # it from 1000/e2e when the session lacks realized_fps_scalar (older
        # profiler sessions, e.g. 1.3.2 detection runs, omit it): a 1000/e2e
        # value is a serial per-frame estimate, not pipelined throughput, and
        # silently substituting it makes EdgeFirst look un-pipelined (every
        # "overlap" collapses to 1.0 by construction). Leave it absent so
        # callers can tell "no throughput recorded" from a real number.
        # NOTE: even when present this is the VALIDATION run's throughput, which
        # is not the edgefirst-profiler benchmark — use the profiler bench
        # (hailortcli / bench-internal) for authoritative performance numbers.
        if m.realized_fps is not None:
            result["fps_pipeline"] = m.realized_fps

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch EdgeFirst Studio validation metrics for a session",
    )
    parser.add_argument("session_id", help="Validation session ID, e.g. v-1a8f")
    parser.add_argument("--raw", action="store_true",
                        help="Print the raw JSON-RPC metrics response and exit")
    args = parser.parse_args()

    if args.raw:
        print(json.dumps(get_raw_metrics(args.session_id), indent=2))
        return

    m = fetch_studio_metrics(args.session_id)
    print(f"Session : {m.session_name} ({m.session_id})")

    if m.det_map_50_95 is not None or m.det_map_50 is not None:
        print("\nDetection:")
        if m.det_map_50_95 is not None:
            print(f"  mAP@0.5:0.95  {m.det_map_50_95 * 100:.2f}%")
        if m.det_map_50 is not None:
            print(f"  mAP@0.5       {m.det_map_50 * 100:.2f}%")
        if m.det_map_75 is not None:
            print(f"  mAP@0.75      {m.det_map_75 * 100:.2f}%")

    if m.seg_map_50_95 is not None or m.seg_map_50 is not None:
        print("\nSegmentation (COCO mAP):")
        if m.seg_map_50_95 is not None:
            print(f"  mAP@0.5:0.95  {m.seg_map_50_95 * 100:.2f}%")
        if m.seg_map_50 is not None:
            print(f"  mAP@0.5       {m.seg_map_50 * 100:.2f}%")
        if m.seg_map_75 is not None:
            print(f"  mAP@0.75      {m.seg_map_75 * 100:.2f}%")

    if m.avg_inference_ms is not None:
        e2e = m.latency_mean_ms or (
            (m.avg_input_ms or 0) + (m.avg_inference_ms or 0) + (m.avg_output_ms or 0)
        )
        print("\nTiming (ms/frame, on-target):")
        if m.avg_input_ms is not None:
            print(f"  preprocess    {m.avg_input_ms:.2f}  (capture + letterbox + normalize)")
        print(f"  inference     {m.avg_inference_ms:.2f}")
        if m.avg_output_ms is not None:
            print(f"  postprocess   {m.avg_output_ms:.2f}")
        print(f"  e2e latency   {e2e:.2f}")
        if m.realized_fps is not None:
            print(f"  realized fps  {m.realized_fps:.1f}  (pipelined throughput)")


if __name__ == "__main__":
    main()
