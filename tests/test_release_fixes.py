"""Regression tests for release-hardening fixes (workstreams B/C/D)."""
from __future__ import annotations

import os

import pytest

from yolo_validator.backends import infer_e2e


# --- D4: pycocotools -1.0 no-data sentinel renders as N/A -------------------
def test_safe_na_negative_is_na():
    from benchmarks.benchmark_a import _safe_na
    assert _safe_na(-1.0) == "N/A"
    assert _safe_na(None) == "N/A"
    assert _safe_na(0.0) == "0.000"
    assert _safe_na(0.3732) == "0.373"


# --- D7: markdown separator always matches the header's pipe layout ---------
def test_markdown_separator_matches_header():
    from benchmarks.benchmark_a import _build_markdown_table
    r = {
        "ult-onnx": {"bbox": {"AP": 0.37, "AP50": 0.52},
                     "timing": {"preprocess": 2.0, "inference": 10.0,
                                "postprocess": 20.0, "e2e": 32.0},
                     "wall_s": 150.0, "fps_wall": 33.3},
        "yv-numpy": {"bbox": {"AP": 0.369, "AP50": 0.515},
                     "timing": {"preprocess": 2.1, "inference": 10.2,
                                "postprocess": -1.0, "e2e": 33.0},
                     "wall_s": 154.0, "fps_wall": 32.4},
    }
    md = _build_markdown_table(r, "demo")
    lines = md.splitlines()
    header = next(line for line in lines if line.startswith("| Config "))
    sep = lines[lines.index(header) + 1]
    assert header.count("|") == sep.count("|")
    # the -1.0 postprocess must render as N/A, not -1.000
    assert "-1.0" not in md


# --- B4: E2E auto-detection is robust to transposed / low-class classic heads
@pytest.mark.parametrize("shapes,expected", [
    ([(1, 84, 8400)], False),                  # classic detect
    ([(1, 116, 8400), (1, 32, 160, 160)], False),  # classic seg + proto
    ([(1, 8400, 84)], False),                  # transposed classic
    ([(1, 8400, 5)], False),                   # transposed low-class classic
    ([(1, 300, 6)], True),                     # e2e detect
    ([(1, 300, 38), (1, 32, 160, 160)], True),  # e2e seg + proto
])
def test_infer_e2e(shapes, expected):
    assert infer_e2e(shapes) is expected


def test_infer_e2e_override_wins():
    assert infer_e2e([(1, 300, 6)], override=False) is False
    assert infer_e2e([(1, 84, 8400)], override=True) is True


# --- B5/_has_end2end: value-of-end2end, not hasattr (needs real models) -----
@pytest.mark.parametrize("model,expected", [("yolov8n.pt", False), ("yolo26n.pt", True)])
def test_has_end2end_real_models(model, expected):
    pytest.importorskip("ultralytics")
    if not os.path.exists(model):
        pytest.skip(f"{model} not present")
    from benchmarks.benchmark_a import _has_end2end
    assert _has_end2end(model) is expected


# --- D1: CLI refuses to run when images <= warmup (was ZeroDivisionError) ---
def test_cli_guard_few_images(tmp_path):
    # one image, warmup 3 -> must SystemExit before any division, and before
    # needing a real model (image listing/guard happens first).
    img = tmp_path / "000001.jpg"
    img.write_bytes(b"not-a-real-jpg")
    from yolo_validator.cli import main
    with pytest.raises(SystemExit):
        main(["--model", "missing.onnx", "--images", str(tmp_path),
              "--warmup", "3"])
