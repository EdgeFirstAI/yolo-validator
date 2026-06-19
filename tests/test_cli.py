from yolo_validator.cli import build_parser


def test_cli_defaults_match_ultralytics_val():
    args = build_parser().parse_args(["--model", "m.onnx", "--images", "imgs/"])
    assert args.conf == 0.001
    assert args.iou == 0.7
    assert args.max_det == 300
    assert args.warmup == 3
    assert args.preprocess == "auto"
    assert args.task == "auto"


def test_cli_requires_model_and_images():
    import pytest
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
