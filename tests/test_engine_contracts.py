"""Tests for engine package contracts and diagnostics."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from floodsr.engine.base import EngineBase
from floodsr.engine.providers import get_onnxruntime_info, get_rasterio_info
from floodsr.engine.pcraster_check import _check_pcraster, get_pcraster_info


@pytest.mark.parametrize(
    "info_getter, required_key",
    [
        pytest.param(get_onnxruntime_info, "available_providers", id="provider_diag_onnxruntime"),
        pytest.param(get_rasterio_info, "version", id="provider_diag_rasterio"),
        pytest.param(get_pcraster_info, "spreadzone_available", id="provider_diag_pcraster"),
    ],
)
@pytest.mark.fast
def test_engine_provider_diagnostics_shape(info_getter, required_key: str):
    """Ensure provider diagnostics return expected keys."""
    info = info_getter()
    assert isinstance(info.get("installed"), bool)
    assert required_key in info


@pytest.mark.fast
def test_check_pcraster_raises_helpful_error_when_missing(monkeypatch: pytest.MonkeyPatch):
    """Ensure the lazy PCRaster import guard explains the supported install path."""
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "pcraster":
            raise ImportError("No module named 'pcraster'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError, match="CostGrow_Terrain"):
        _check_pcraster()


@pytest.mark.fast
def test_engine_base_is_abstract():
    """Ensure EngineBase cannot be instantiated directly."""
    with pytest.raises(TypeError):
        EngineBase()


@pytest.mark.fast
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


