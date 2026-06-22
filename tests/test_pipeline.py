# tests/test_pipeline.py
import numpy as np
from yolo_validator.pipeline import ValidationPipeline


def test_pipeline_runs_one_image(rgb_image, mock_backend):
    path, w, h = rgb_image
    pipe = ValidationPipeline(mock_backend, preprocess_path="numpy",
                              postprocess_path="numpy",
                              conf=0.001, iou=0.7, max_det=300, with_masks=True)
    result = pipe.infer_one(path)
    assert len(result.detections.scores) == 1
    # timing has all the canonical fine stages
    for k in ("load_decode", "preprocess", "inference", "decode", "mask"):
        assert k in result.timings and result.timings[k] >= 0.0


def test_pipeline_warmup_excluded_from_collection(rgb_image, mock_backend):
    path, w, h = rgb_image
    pipe = ValidationPipeline(mock_backend, preprocess_path="numpy",
                              postprocess_path="numpy",
                              conf=0.001, iou=0.7, max_det=300, with_masks=False)
    stats = pipe.run([path, path, path], warmup=2)
    # 3 images, 2 warmup -> 1 measured sample per stage
    assert stats.stages["inference"].count == 1


def _collect(pipe, paths, batch_size):
    out = []
    pipe.run(paths, warmup=0, batch_size=batch_size, on_frame=lambda r: out.append(
        (r.detections.boxes.copy(), r.detections.scores.copy(),
         r.detections.classes.copy())))
    return out


def test_batch_matches_single_image(rgb_image, mock_backend):
    """batch_size=N must yield byte-identical predictions to batch_size=1."""
    path, w, h = rgb_image
    paths = [path] * 5
    pipe = ValidationPipeline(mock_backend, preprocess_path="numpy",
                              postprocess_path="numpy", with_masks=False)
    single = _collect(pipe, paths, 1)
    batched = _collect(pipe, paths, 2)     # chunks [0:2],[2:4],[4:5] (last b=1)
    assert len(single) == len(batched) == 5
    for (b1, s1, c1), (b2, s2, c2) in zip(single, batched):
        np.testing.assert_array_equal(b1, b2)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(c1, c2)


def test_runstats_batch_size_propagates(rgb_image, mock_backend):
    path, w, h = rgb_image
    pipe = ValidationPipeline(mock_backend, preprocess_path="numpy",
                              postprocess_path="numpy", with_masks=False)
    stats = pipe.run([path] * 4, warmup=0, batch_size=4)
    assert stats.batch_size == 4
    assert stats.n_images == 4
    # batch-amortized inference still recorded once per image
    assert stats.stages["inference"].count == 4


def test_batch_guard_rejects_fixed_batch_model(rgb_image):
    """batch_size>1 on a fixed batch=1 model raises a clear error."""
    import pytest
    from yolo_validator.backends import ModelSpec

    class FixedBatchBackend:
        def __init__(self):
            self.spec = ModelSpec(640, 640, "detect")
            self.input_batch = 1   # static batch=1 graph
        def run(self, x):
            return [np.zeros((x.shape[0], 84, 8400), np.float32)], None

    path, w, h = rgb_image
    pipe = ValidationPipeline(FixedBatchBackend(), preprocess_path="numpy",
                              postprocess_path="numpy", with_masks=False)
    with pytest.raises(ValueError, match="fixed"):
        pipe.run([path] * 4, warmup=0, batch_size=4)
