"""Tests for engine package exports."""

from floodsr import engine


def test_engine_package_exports_have_expected_symbols():
    """Ensure engine package exports ORT engine and provider helpers."""
    assert hasattr(engine, "EngineORT")
    assert hasattr(engine, "get_onnxruntime_info")

