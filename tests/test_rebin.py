# tests/test_rebin.py
"""TDD tests for benchmarks/rebin.py."""
import pytest
from benchmarks.rebin import rebin_frame, rebin_samples, rebin_ultralytics


def test_rebin_frame_basic():
    timings = {"load_decode": 5.0, "preprocess": 2.0, "inference": 10.0, "decode": 1.5, "mask": 0.5}
    r = rebin_frame(timings)
    assert r["preprocess"] == pytest.approx(7.0)     # load_decode + letterbox = 5 + 2
    assert r["inference"] == 10.0
    assert r["postprocess"] == pytest.approx(2.0)   # decode + mask = 1.5 + 0.5
    assert r["e2e"] == pytest.approx(19.0)           # decode + pre + inf + decode + mask
    assert "load_decode" not in r                    # folded into preprocess, not a separate key


def test_rebin_frame_no_mask():
    """Detect model: no mask stage — mask treated as 0."""
    timings = {"load_decode": 5.0, "preprocess": 2.0, "inference": 10.0, "decode": 1.5}
    r = rebin_frame(timings)
    assert r["preprocess"] == pytest.approx(7.0)     # 5 + 2
    assert r["postprocess"] == pytest.approx(1.5)
    assert r["e2e"] == pytest.approx(18.5)           # 7 + 10 + 1.5


def test_rebin_frame_includes_load_decode():
    """JPEG decode is folded into preprocess (and therefore e2e), not dropped."""
    timings = {"load_decode": 99.0, "preprocess": 1.0, "inference": 5.0, "decode": 1.0}
    r = rebin_frame(timings)
    assert "load_decode" not in r                    # folded into preprocess
    assert r["preprocess"] == pytest.approx(100.0)   # 99 + 1
    assert r["e2e"] == pytest.approx(106.0)          # 100 + 5 + 1


def test_rebin_frame_empty_timings():
    """All missing stages → all zeros."""
    r = rebin_frame({})
    assert r["preprocess"] == 0.0
    assert r["inference"] == 0.0
    assert r["postprocess"] == 0.0
    assert r["e2e"] == 0.0


def test_rebin_samples_aggregates():
    frames = [
        {"preprocess": 1.0, "inference": 10.0, "postprocess": 2.0, "e2e": 13.0},
        {"preprocess": 3.0, "inference": 20.0, "postprocess": 4.0, "e2e": 27.0},
    ]
    stats = rebin_samples(frames)
    assert stats["preprocess"].mean_ms == pytest.approx(2.0)
    assert stats["inference"].mean_ms == pytest.approx(15.0)
    assert stats["postprocess"].mean_ms == pytest.approx(3.0)
    assert stats["e2e"].mean_ms == pytest.approx(20.0)


def test_rebin_samples_from_raw_pipeline_timings():
    """rebin_samples correctly handles raw pipeline timing dicts (with load_decode/mask keys)."""
    frames = [
        {"load_decode": 5.0, "preprocess": 2.0, "inference": 10.0, "decode": 1.5, "mask": 0.5},
        {"load_decode": 6.0, "preprocess": 3.0, "inference": 12.0, "decode": 2.0, "mask": 1.0},
    ]
    stats = rebin_samples(frames)
    # postprocess = decode + mask per frame, then mean
    # frame 0: post=2.0, frame 1: post=3.0 → mean=2.5
    assert stats["postprocess"].mean_ms == pytest.approx(2.5)
    # preprocess = load_decode + preprocess per frame: 7.0, 9.0 → mean=8.0
    assert stats["preprocess"].mean_ms == pytest.approx(8.0)
    # e2e per frame: 7+10+2=19.0, 9+12+3=24.0 → mean=21.5
    assert stats["e2e"].mean_ms == pytest.approx(21.5)
    assert stats["inference"].mean_ms == pytest.approx(11.0)


def test_rebin_samples_requires_frames():
    with pytest.raises(ValueError):
        rebin_samples([])


def test_rebin_samples_count():
    frames = [{"preprocess": 1.0, "inference": 5.0, "decode": 1.0} for _ in range(5)]
    stats = rebin_samples(frames)
    assert stats["preprocess"].count == 5


def test_rebin_ultralytics_basic():
    speed = {"preprocess": 1.5, "inference": 8.0, "postprocess": 2.5}
    result = rebin_ultralytics(speed, n_images=100)
    assert result["preprocess"] == pytest.approx(1.5)
    assert result["inference"] == pytest.approx(8.0)
    assert result["postprocess"] == pytest.approx(2.5)
    assert result["e2e"] == pytest.approx(12.0)


def test_rebin_ultralytics_missing_keys():
    """Missing keys default to 0.0."""
    speed = {"inference": 10.0}
    result = rebin_ultralytics(speed, n_images=50)
    assert result["preprocess"] == 0.0
    assert result["postprocess"] == 0.0
    assert result["e2e"] == pytest.approx(10.0)


def test_rebin_frame_canonical_dict_postprocess_included_in_e2e():
    """When dict has 'postprocess' key (no decode/mask), e2e must include postprocess."""
    timings = {"preprocess": 2.0, "inference": 10.0, "postprocess": 3.0}
    r = rebin_frame(timings)
    assert r["postprocess"] == pytest.approx(3.0)
    assert r["e2e"] == pytest.approx(15.0)  # 2+10+3, NOT 2+10+0+0
