"""Tests for engine base abstractions."""

from pathlib import Path

import numpy as np
import pytest

from floodsr.engine.base import EngineBase


def test_engine_base_is_abstract():
    """Ensure EngineBase cannot be instantiated directly."""
    with pytest.raises(TypeError) as exc_info:
        EngineBase()
    assert isinstance(exc_info.value, TypeError)
    assert "abstract" in str(exc_info.value).lower()


def test_engine_base_contract_with_dummy_subclass():
    """Ensure subclass implementing the abstract API can run and return arrays."""

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

    engine = DummyEngine()
    result = engine.run_tile(np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32))
    assert result["prediction_m"].dtype == np.float32
    assert result["prediction_m"].size > 0

