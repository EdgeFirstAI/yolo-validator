"""Pure-logic tests for benchmarks/qnn_infer.py (no ultralytics / QNN needed)."""
import pytest

from benchmarks.qnn_infer import ValPhaseTimer, _safe_label


class _Clock:
    def __init__(self, ticks):
        self._ticks = iter(ticks)

    def __call__(self):
        return next(self._ticks)


def test_phase_timer_splits_load_and_batch():
    # val_start=10.0; batch 1 starts 10.005, ends 10.025; batch 2 starts 10.031,
    # ends 10.052; val_end=10.060.
    timer = ValPhaseTimer(clock=_Clock([10.0, 10.005, 10.025, 10.031, 10.052, 10.060]))
    cb = timer.callbacks
    cb["on_val_start"](None)
    for _ in range(2):
        cb["on_val_batch_start"](None)
        cb["on_val_batch_end"](None)
    cb["on_val_end"](None)

    assert timer.load_ms == pytest.approx([5.0, 6.0])
    assert timer.batch_ms == pytest.approx([20.0, 21.0])

    s = timer.summary(t_call=8.0, t_done=11.0)
    assert s["image_load_ms"]["count"] == 2
    assert s["image_load_ms"]["mean_ms"] == pytest.approx(5.5)
    assert s["batch_ms"]["max_ms"] == pytest.approx(21.0)
    assert s["val_phases_s"] == pytest.approx({"setup": 2.0, "loop": 0.06, "finalize": 0.94})


def test_phase_timer_registers_all_val_events():
    assert set(ValPhaseTimer().callbacks) == {
        "on_val_start", "on_val_batch_start", "on_val_batch_end", "on_val_end"}


@pytest.mark.parametrize("bad", ["../x", "a b", "-x", ""])
def test_safe_label_rejects(bad):
    with pytest.raises(SystemExit):
        _safe_label(bad)


def test_safe_label_accepts():
    assert _safe_label("yolo26n-seg-classic") == "yolo26n-seg-classic"


@pytest.mark.parametrize("bad", ["../x", "a/b", "..", "-x", ""])
def test_export_rejects_unsafe_model_names(bad):
    from benchmarks.export_qnn import _safe_model

    with pytest.raises(SystemExit):
        _safe_model(bad)
