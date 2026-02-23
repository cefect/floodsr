"""Tests for engine package contracts and diagnostics."""

from pathlib import Path

import numpy as np
import pytest

from floodsr.engine import EngineORT
from floodsr.engine.base import EngineBase
from floodsr.engine.providers import get_onnxruntime_info, get_rasterio_info


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "info_getter, required_key",
    [
        pytest.param(get_onnxruntime_info, "available_providers", id="provider_diag_onnxruntime"),
        pytest.param(get_rasterio_info, "version", id="provider_diag_rasterio"),
    ],
)
def test_engine_provider_diagnostics_shape(info_getter, required_key: str):
    """Ensure provider diagnostics return expected keys."""
    info = info_getter()
    assert isinstance(info.get("installed"), bool)
    assert required_key in info


def test_engine_base_is_abstract():
    """Ensure EngineBase cannot be instantiated directly."""
    with pytest.raises(TypeError):
        EngineBase()


def test_engine_base_contract_with_dummy_subclass():
    """Ensure a concrete EngineBase subclass returns prediction arrays."""

    class DummyEngine(EngineBase):
        """Simple concrete engine for contract tests."""

        def __init__(self):
            """Initialize dummy state."""
            self._model_fp = Path("dummy.onnx")

        def load(self) -> None:
            """No-op load for dummy engine."""

        def run_tile(self, depth_lr_m: np.ndarray, dem_hr_m: np.ndarray, **kwargs):
            """Return a simple prediction payload."""
            return {"prediction_m": np.asarray(dem_hr_m, dtype=np.float32)}

        def model_path(self) -> Path:
            """Return a dummy path."""
            return self._model_fp

    engine_instance = DummyEngine()
    result = engine_instance.run_tile(np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32))
    assert result["prediction_m"].dtype == np.float32
    assert result["prediction_m"].size > 0


@pytest.mark.parametrize(
    "repeat_run",
    [
        pytest.param(False, id="ort_contract_single_run"),
        pytest.param(True, id="ort_contract_repeat_run_is_deterministic"),
    ],
)
def test_engine_ort_run_tile_contract(tohr_model_fp, ort_tile_inputs, logger, repeat_run: bool):
    """Ensure ORT predictions are float32, non-empty, and deterministic on repeat."""
    pytest.importorskip("onnxruntime")
    engine_instance = EngineORT(tohr_model_fp, logger=logger)
    run1 = engine_instance.run_tile(
        ort_tile_inputs["depth_lr"],
        ort_tile_inputs["dem_hr"],
        depth_lr_nodata=ort_tile_inputs["depth_lr_nodata"],
        dem_hr_nodata=ort_tile_inputs["dem_hr_nodata"],
        logger=logger,
    )
    assert run1["prediction_m"].dtype == np.float32
    assert run1["prediction_m"].size > 0

    if repeat_run:
        run2 = engine_instance.run_tile(
            ort_tile_inputs["depth_lr"],
            ort_tile_inputs["dem_hr"],
            depth_lr_nodata=ort_tile_inputs["depth_lr_nodata"],
            dem_hr_nodata=ort_tile_inputs["dem_hr_nodata"],
            logger=logger,
        )
        assert isinstance(run2["prediction_m"], np.ndarray)
        assert np.array_equal(run1["prediction_m"], run2["prediction_m"])
