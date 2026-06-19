"""Regression guard for benchmark variant expansion.

yolov8 (natively classic) must yield ONE classic variant; yolo26 (natively
NMS-free) must yield BOTH a classic and an nmsfree variant. The discriminator
is the model's native ``end2end`` value, not mere attribute existence —
see benchmarks.benchmark_a._has_end2end.
"""
from pathlib import Path

import benchmarks.benchmark_a as ba


def test_variants_classic_only(monkeypatch):
    monkeypatch.setattr(ba, "_has_end2end", lambda p: False)
    monkeypatch.setattr(ba, "_infer_task", lambda n: "detect")
    variants = ba._get_variants("yolov8n", Path("yolov8n.pt"))
    assert len(variants) == 1
    label, _pt, task, mode = variants[0]
    assert label == "yolov8n"
    assert task == "detect"
    assert mode == "classic"


def test_variants_dual_mode(monkeypatch):
    monkeypatch.setattr(ba, "_has_end2end", lambda p: True)
    monkeypatch.setattr(ba, "_infer_task", lambda n: "segment")
    variants = ba._get_variants("yolo26n-seg", Path("yolo26n-seg.pt"))
    assert [(v[0], v[3]) for v in variants] == [
        ("yolo26n-seg-classic", "classic"),
        ("yolo26n-seg-nmsfree", "nmsfree"),
    ]
    assert all(v[2] == "segment" for v in variants)
