"""Tests for ORT engine behavior."""

import numpy as np
import pytest

from floodsr.engine import EngineORT
from tests.conftest import inference_model_fp, tile_data_dir


@pytest.fixture(scope="function")
def tile_arrays(tile_data_dir):
    """Load low-res depth and high-res DEM arrays for engine tests."""
    rasterio = pytest.importorskip("rasterio")
    with rasterio.open(tile_data_dir / "lowres032.tif") as ds:
        depth_lr_raw = ds.read(1).astype(np.float32)
        depth_lr_nodata = ds.nodata
    with rasterio.open(tile_data_dir / "hires002_dem.tif") as ds:
        dem_hr_raw = ds.read(1).astype(np.float32)
        dem_hr_nodata = ds.nodata
    return {
        "depth_lr_raw": depth_lr_raw,
        "dem_hr_raw": dem_hr_raw,
        "depth_lr_nodata": depth_lr_nodata,
        "dem_hr_nodata": dem_hr_nodata,
    }


def test_engine_ort_run_tile_returns_float32_prediction(inference_model_fp, tile_arrays):
    """Ensure ORT engine inference returns float32 predictions with data."""
    pytest.importorskip("onnxruntime")
    engine = EngineORT(inference_model_fp)
    result = engine.run_tile(
        tile_arrays["depth_lr_raw"],
        tile_arrays["dem_hr_raw"],
        depth_lr_nodata=tile_arrays["depth_lr_nodata"],
        dem_hr_nodata=tile_arrays["dem_hr_nodata"],
    )
    assert result["prediction_m"].dtype == np.float32
    assert result["prediction_m"].size > 0


def test_engine_ort_run_tile_is_deterministic(inference_model_fp, tile_arrays):
    """Ensure repeated ORT inference on same tile returns identical arrays."""
    pytest.importorskip("onnxruntime")
    engine = EngineORT(inference_model_fp)
    run1 = engine.run_tile(
        tile_arrays["depth_lr_raw"],
        tile_arrays["dem_hr_raw"],
        depth_lr_nodata=tile_arrays["depth_lr_nodata"],
        dem_hr_nodata=tile_arrays["dem_hr_nodata"],
    )
    run2 = engine.run_tile(
        tile_arrays["depth_lr_raw"],
        tile_arrays["dem_hr_raw"],
        depth_lr_nodata=tile_arrays["depth_lr_nodata"],
        dem_hr_nodata=tile_arrays["dem_hr_nodata"],
    )
    assert isinstance(run1["prediction_m"], np.ndarray)
    assert np.array_equal(run1["prediction_m"], run2["prediction_m"])
