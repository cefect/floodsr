"""ToHR pipeline entrypoint."""

import logging
import tempfile
from pathlib import Path

from floodsr.model_registry import resolve_model_worker_class
from floodsr.preprocessing import write_platform_prepared_rasters


def tohr(
    model_version: str,
    model_fp: str | Path,
    depth_lr_fp: str | Path,
    dem_hr_fp: str | Path,
    output_fp: str | Path,
    crs_policy: str = "strict",
    max_depth: float | None = None,
    dem_pct_clip: float | None = None,
    window_method: str = "feather",
    tile_overlap: int | None = None,
    tile_size: int | None = None,
    logger=None,
) -> dict[str, object]:
    """Run one ToHR pass through the model worker lifecycle."""
    log = logger or logging.getLogger(__name__)
    assert model_version, "model_version cannot be empty"
    model_path = Path(model_fp).expanduser().resolve()
    assert model_path.exists(), f"model file does not exist: {model_path}"

    # Resolve worker and run inside context-managed lifecycle.
    worker_class = resolve_model_worker_class(model_version)
    worker = worker_class(model_fp=model_path, logger=log)
    with worker as ready_worker:
        # Platform-level preprocessing runs before model worker execution.
        with tempfile.TemporaryDirectory(prefix="floodsr-platform-prep-") as prepped_dir:
            platform = write_platform_prepared_rasters(
                depth_lr_fp,
                dem_hr_fp,
                prepped_dir,
                crs_policy=crs_policy,
                logger=log,
            )
            result = ready_worker.run(
                depth_lr_fp=platform["depth_lr_prepared_fp"],
                dem_hr_fp=platform["dem_hr_prepared_fp"],
                output_fp=output_fp,
                crs_policy=crs_policy,
                max_depth=max_depth,
                dem_pct_clip=dem_pct_clip,
                window_method=window_method,
                tile_overlap=tile_overlap,
                tile_size=tile_size,
            )
    return result
