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
