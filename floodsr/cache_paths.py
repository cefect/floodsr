"""Cache path helpers for model weights."""

import logging
from pathlib import Path
from platformdirs import user_cache_dir


APP_NAME = "floodsr"
APP_AUTHOR = "floodsr"
log = logging.getLogger(__name__)


def get_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Return a writable cache directory and ensure it exists."""
    # Prefer an explicit cache directory when one is supplied.
    if cache_dir is not None:
        path = Path(cache_dir).expanduser().resolve()
    else:
        # Use a stable platform cache path.
        path = Path(user_cache_dir(APP_NAME, APP_AUTHOR))
    path.mkdir(parents=True, exist_ok=True)
    assert path.exists(), f"failed to create cache directory: {path}"
    log.debug(f"resolved cache directory to\n    {path}")
    return path


def get_model_cache_path(
    model_version: str,
    file_name: str,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return the cache path for a specific model file."""
    assert model_version, "model_version cannot be empty"
    assert file_name, "file_name cannot be empty"

    # Group each model version under its own cache subdirectory.
    model_fp = get_cache_dir(cache_dir) / model_version / file_name
    model_fp.parent.mkdir(parents=True, exist_ok=True)
    log.debug(f"resolved model cache path to\n    {model_fp}")
    return model_fp
