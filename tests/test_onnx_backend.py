import os
import pytest

MODEL = os.environ.get("YOLO_VALIDATOR_TEST_ONNX")


@pytest.mark.skipif(not MODEL, reason="set YOLO_VALIDATOR_TEST_ONNX to a yolo*-seg.onnx to run")
def test_onnx_backend_loads_and_infers(rgb_image):
    from yolo_validator.backends import load_backend
    from yolo_validator.pipeline import ValidationPipeline

    path, w, h = rgb_image
    backend = load_backend(MODEL, runtime="onnx", provider="cpu")
    assert backend.spec.input_w > 0 and backend.spec.task in ("detect", "segment")
    pipe = ValidationPipeline(backend, preprocess_path="numpy", postprocess_path="numpy",
                              with_masks=(backend.spec.task == "segment"))
    res = pipe.infer_one(path)
    assert "inference" in res.timings


def test_onnx_backend_importable_without_model():
    # the module imports even when onnxruntime missing is handled at call time
    import yolo_validator.backends.onnxruntime_backend as m
    assert hasattr(m, "OnnxRuntimeBackend")
