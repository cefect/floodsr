"""Inference engine interfaces for FloodSR."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class EngineBase(ABC):
    """Abstract interface for model inference engines."""

    @abstractmethod
    def load(self) -> None:
        """Load model resources into memory."""

    @abstractmethod
    def run_tile(
        self,
        depth_lr_m: np.ndarray,
        dem_hr_m: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run one inference pass for a low-res depth + high-res DEM pair."""

    @abstractmethod
    def model_path(self) -> Path:
        """Return the model path used by this engine."""

