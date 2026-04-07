"""Shared helpers for model-specific tests."""


def assert_hard_only_windowed_selection(worker, small_shape=(2048, 2048), large_shape=(4096, 4096)) -> None:
    """Assert the shared hard-only large-raster execution selector contract."""
    assert worker._resolve_execution_path("hard", small_shape) == "simple"
    assert worker._resolve_execution_path("feather", small_shape) == "simple"
    assert worker._resolve_execution_path("hard", large_shape) == "windowed"
    assert worker._resolve_execution_path("feather", large_shape) == "simple"


def assert_result_raster_contract(result: dict, expected_shape=None):
    """Assert a FloodSR raster result exists, is float32, and is non-empty."""
    import numpy as np
    import rasterio

    with rasterio.open(result["output_fp"]) as ds:
        pred = ds.read(1)
    if expected_shape is not None:
        assert pred.shape == expected_shape
    assert pred.dtype == np.float32
    assert pred.size > 0
    return pred
