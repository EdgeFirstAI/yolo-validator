import pytest
from yolo_validator._stats import stage_stats, StageStats


def test_stage_stats_percentiles_no_trim():
    samples = [float(x) for x in range(1, 101)]  # 1..100 ms
    s = stage_stats(samples)
    assert s.count == 100
    assert s.min_ms == 1.0
    assert s.max_ms == 100.0           # NOT trimmed — the 100 ms outlier survives
    assert s.mean_ms == pytest.approx(50.5)
    assert s.p50_ms == pytest.approx(50.5, abs=0.5)
    assert s.p95_ms == pytest.approx(95.05, abs=0.5)
    assert s.p99_ms == pytest.approx(99.01, abs=0.5)


def test_stage_stats_single_sample():
    s = stage_stats([7.0])
    assert s.count == 1 and s.min_ms == s.max_ms == s.p99_ms == 7.0


def test_stage_stats_empty_raises():
    with pytest.raises(ValueError):
        stage_stats([])
