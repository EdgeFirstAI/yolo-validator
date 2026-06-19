from yolo_validator.backends import ModelSpec, DeviceTiming


def test_modelspec_fields():
    spec = ModelSpec(input_w=640, input_h=640, task="segment")
    assert spec.input_w == 640 and spec.task == "segment"


def test_devicetiming_total():
    dt = DeviceTiming(dma_input_ms=1.0, compute_ms=4.0, dma_output_ms=2.0)
    assert dt.total_ms == 7.0
