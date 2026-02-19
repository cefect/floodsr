"""Tests for engine provider diagnostics."""

from floodsr.engine.providers import get_onnxruntime_info, get_rasterio_info


def test_get_onnxruntime_info_shape():
    """Ensure ORT diagnostic payload has expected keys and provider list dtype."""
    info = get_onnxruntime_info()
    assert isinstance(info.get("available_providers"), list)
    assert "installed" in info


def test_get_rasterio_info_shape():
    """Ensure rasterio diagnostic payload has expected installation keys."""
    info = get_rasterio_info()
    assert isinstance(info.get("installed"), bool)
    assert "version" in info

