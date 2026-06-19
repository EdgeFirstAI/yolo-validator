"""Hermetic tests for the TensorRT engine loader's metadata handling.

The TRT runtime itself needs a GPU, but the Ultralytics-metadata stripping
is pure byte logic and is tested here without TensorRT installed.
"""
import json

from yolo_validator.backends.tensorrt_backend import strip_ultralytics_metadata


def test_strip_ultralytics_header():
    meta = json.dumps({"stride": 32, "names": {"0": "person"}}).encode("utf-8")
    engine = b"\x00RAW-ENGINE-BYTES\xff" * 4
    blob = len(meta).to_bytes(4, byteorder="little", signed=True) + meta + engine
    assert strip_ultralytics_metadata(blob) == engine


def test_raw_engine_passthrough():
    # No valid 4-byte-length + JSON header -> returned unchanged.
    raw = b"\x12\x34\x56\x78" + b"not-json-engine-bytes" * 8
    assert strip_ultralytics_metadata(raw) == raw


def test_short_input_passthrough():
    assert strip_ultralytics_metadata(b"ab") == b"ab"
