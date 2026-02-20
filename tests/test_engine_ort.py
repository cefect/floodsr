"""Tests for ORT engine behavior."""

import numpy as np
import pytest

from floodsr.engine import EngineORT
from tests.conftest import inference_model_fp, ort_tile_inputs


def test_engine_ort_run_tile_returns_float32_prediction(inference_model_fp, ort_tile_inputs):
    """Ensure ORT engine inference returns float32 predictions with data."""
    pytest.importorskip("onnxruntime")
    engine = EngineORT(inference_model_fp)
    result = engine.run_tile(
        ort_tile_inputs["depth_lr"],
        ort_tile_inputs["dem_hr"],
        depth_lr_nodata=ort_tile_inputs["depth_lr_nodata"],
        dem_hr_nodata=ort_tile_inputs["dem_hr_nodata"],
    )
    assert result["prediction_m"].dtype == np.float32
    assert result["prediction_m"].size > 0


def test_engine_ort_run_tile_is_deterministic(inference_model_fp, ort_tile_inputs):
    """Ensure repeated ORT inference on same tile returns identical arrays."""
    pytest.importorskip("onnxruntime")
    engine = EngineORT(inference_model_fp)
    run1 = engine.run_tile(
        ort_tile_inputs["depth_lr"],
        ort_tile_inputs["dem_hr"],
        depth_lr_nodata=ort_tile_inputs["depth_lr_nodata"],
        dem_hr_nodata=ort_tile_inputs["dem_hr_nodata"],
    )
    run2 = engine.run_tile(
        ort_tile_inputs["depth_lr"],
        ort_tile_inputs["dem_hr"],
        depth_lr_nodata=ort_tile_inputs["depth_lr_nodata"],
        dem_hr_nodata=ort_tile_inputs["dem_hr_nodata"],
    )
    assert isinstance(run1["prediction_m"], np.ndarray)
    assert np.array_equal(run1["prediction_m"], run2["prediction_m"])
