"""Base model worker contract for FloodSR model modules."""

import logging
from pathlib import Path
from typing import Any


class Model:
    """Base class for model workers."""

    model_version = ""

    def __init__(self, model_fp: str | Path, *, model_version: str | None = None, logger=None):
        """Initialize a model worker with artifact path and logger."""
        self.model_fp = Path(model_fp).expanduser().resolve()
        assert self.model_fp.exists(), f"model file does not exist: {self.model_fp}"
        self.log = logger or logging.getLogger(__name__)
        if model_version is not None:
            assert model_version, "model_version cannot be empty"
            if self.model_version:
                assert model_version == self.model_version, (
                    f"worker model_version '{self.model_version}' does not match requested '{model_version}'"
                )
            else:
                self.model_version = model_version

    @classmethod
    def is_valid(cls, model_fp: str | Path) -> bool:
        """Return whether this worker can run from the provided artifact path."""
        return Path(model_fp).expanduser().resolve().exists()

    def __enter__(self):
        """Enter worker context."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit worker context."""
        return False

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Run model-specific ToHR flow and return diagnostics."""
        raise NotImplementedError("Model.run must be implemented by subclasses")
