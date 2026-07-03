"""Hermetic tests for the Ara240 lane's pure helpers (no dvapi/hardware)."""
import json

import numpy as np
import pytest

from benchmarks.ara2_infer import (
    STAGE_KEYS, _decode_outputs, _infer_role, _stats,
    _strip_trailing_ones, _to_nc, _merge_partials, _build_doc,
)
from yolo_validator.coco_output import coco80_to_coco91
from yolo_validator.letterbox import LetterboxInfo


def test_stats_no_trimming():
    vals = [1.0] * 99 + [101.0]          # one outlier must NOT be trimmed
    s = _stats(vals)
    assert s["max"] == 101.0
    assert s["n"] == 100
    assert s["mean"] == pytest.approx(2.0)
    for k in ("min", "p50", "p95", "p99"):
        assert k in s


def test_stats_nan_filtered_and_empty():
    s = _stats([1.0, float("nan"), 3.0])
    assert s["n"] == 2 and s["mean"] == pytest.approx(2.0)
    assert _stats([]) is None
    assert _stats([float("nan")]) is None


def test_strip_trailing_ones():
    assert _strip_trailing_ones([4, 8400, 1]) == [4, 8400]
    assert _strip_trailing_ones([32, 160, 160]) == [32, 160, 160]


def test_infer_role():
    seen = set()
    assert _infer_role([32, 160, 160], seen) == "protos"
    assert _infer_role([4, 8400], seen) == "boxes"
    assert _infer_role([80, 8400], seen) == "scores"
    assert _infer_role([32, 8400], seen) == "mask_coefs"
    assert _infer_role([2, 8400], seen) == "box_xy"
    seen.add("box_xy")
    assert _infer_role([2, 8400], seen) == "box_wh"


def test_to_nc_orientations():
    a = np.arange(8, dtype=np.float32).reshape(2, 4)      # (C=2, N=4)
    out = _to_nc(a)
    assert out.shape == (4, 2)
    b = np.arange(8, dtype=np.float32).reshape(1, 2, 4)   # batch dim
    assert _to_nc(b).shape == (4, 2)


def _lb_identity():
    return LetterboxInfo(1.0, 0, 0, 640, 640)


def test_decode_det_single_box():
    N = 100
    boxes = np.zeros((4, N), dtype=np.float32)
    scores = np.zeros((80, N), dtype=np.float32)
    boxes[:, 5] = [320.0, 320.0, 100.0, 200.0]   # xywh, letterbox px
    scores[2, 5] = 0.9
    recs = _decode_outputs({"boxes": boxes, "scores": scores}, 640,
                           _lb_identity(), 42, coco80_to_coco91(),
                           score_th=0.001)
    assert len(recs) == 1
    r = recs[0]
    assert r["image_id"] == 42
    assert r["category_id"] == coco80_to_coco91()[2]
    assert r["score"] == pytest.approx(0.9, abs=1e-5)
    x, y, w, h = r["bbox"]
    assert (x, y, w, h) == pytest.approx((270.0, 220.0, 100.0, 200.0), abs=1.0)


def test_decode_split_box_head():
    """box_xy + box_wh (2,N) halves must decode identically to merged boxes."""
    N = 50
    xy = np.zeros((2, N), dtype=np.float32)
    wh = np.zeros((2, N), dtype=np.float32)
    scores = np.zeros((80, N), dtype=np.float32)
    xy[:, 7] = [320.0, 320.0]
    wh[:, 7] = [100.0, 200.0]
    scores[0, 7] = 0.8
    recs = _decode_outputs({"box_xy": xy, "box_wh": wh, "scores": scores},
                           640, _lb_identity(), 7, coco80_to_coco91(),
                           score_th=0.001)
    assert len(recs) == 1
    assert recs[0]["bbox"] == pytest.approx((270.0, 220.0, 100.0, 200.0),
                                            abs=1.0)


def test_decode_seg_emits_rle():
    N = 100
    boxes = np.zeros((4, N), dtype=np.float32)
    scores = np.zeros((80, N), dtype=np.float32)
    coeffs = np.zeros((32, N), dtype=np.float32)
    protos = np.zeros((32, 160, 160), dtype=np.float32)
    boxes[:, 5] = [320.0, 320.0, 200.0, 200.0]
    scores[0, 5] = 0.95
    coeffs[0, 5] = 5.0            # strong positive coeff on proto ch 0
    protos[0, 60:100, 60:100] = 5.0   # blob inside the box (proto is 1/4 res)
    recs = _decode_outputs({"boxes": boxes, "scores": scores,
                            "mask_coefs": coeffs, "protos": protos},
                           640, _lb_identity(), 9, coco80_to_coco91(),
                           score_th=0.001)
    assert len(recs) == 1
    assert "segmentation" in recs[0]
    assert recs[0]["segmentation"]["counts"]     # non-empty RLE


def test_decode_empty_below_threshold():
    boxes = np.zeros((4, 10), dtype=np.float32)
    scores = np.full((80, 10), 1e-5, dtype=np.float32)
    recs = _decode_outputs({"boxes": boxes, "scores": scores}, 640,
                           _lb_identity(), 1, coco80_to_coco91(),
                           score_th=0.001)
    assert recs == []


def _fake_partial(tmp_path, start, count, wall):
    d = {"start": start, "count": count, "wall_s": wall, "imgsz": 640,
         "preds": [{"image_id": start + i, "category_id": 1,
                    "bbox": [0, 0, 10, 10], "score": 0.5}
                   for i in range(count)],
         "timings": {k: [1.0] * count for k in STAGE_KEYS}}
    p = tmp_path / f"partial_{start:06d}.json"
    p.write_text(json.dumps(d))
    return p


def test_merge_partials(tmp_path):
    p1 = _fake_partial(tmp_path, 0, 3, 1.5)
    p2 = _fake_partial(tmp_path, 3, 2, 1.0)
    merged = _merge_partials([p2, p1])          # order-independent
    assert merged["n_images"] == 5
    assert merged["wall_s"] == pytest.approx(2.5)
    assert len(merged["preds"]) == 5
    assert merged["preds"][0]["image_id"] == 0  # sorted by start
    assert len(merged["timings"]["npu_core"]) == 5


def test_build_doc_schema(tmp_path):
    merged = _merge_partials([_fake_partial(tmp_path, 0, 4, 2.0)])
    merged["meta"] = {"model": "yolov8n", "task": "detect",
                      "vendor": "kinara-sdk-1.2.1",
                      "artifact": "yolov8n-kinara-1.2.1.dvm",
                      "device": "imx95-ara240", "chunks": 1,
                      "chunk_retries": 0,
                      "host": {"machine": "aarch64", "node": "t",
                               "system": "linux", "device": "imx95-ara240"}}
    metrics = {"bbox": {"AP": 0.3, "AP50": 0.5}}
    doc = _build_doc(merged, metrics)
    assert doc["label"] == "yolov8n" and doc["task"] == "detect"
    cfg = doc["configs"]["yv-ara2"]
    assert cfg["bbox"]["AP"] == 0.3
    assert cfg["npu"] == {"h2d": pytest.approx(1.0), "core": pytest.approx(1.0),
                          "d2h": pytest.approx(1.0)}
    assert cfg["timing"]["e2e"] == pytest.approx(1.0)
    assert cfg["fps_wall"] == pytest.approx(2.0)
    assert cfg["vendor"] == "kinara-sdk-1.2.1"
    assert cfg["batch"] == 1 and cfg["n_images"] == 4
    assert cfg["stats"]["inference"]["p99"] == pytest.approx(1.0)
