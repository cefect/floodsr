"""Registry and dispatch for DEM source backends."""

import logging
from importlib import import_module
from pathlib import Path

from floodsr.dem_sources.base import DemFetchResult


_SOURCE_REGISTRY = {
    "hrdem": ("floodsr.dem_sources.hrdem_mosaic", "main_fetch_hrdem_for_lowres_tile"),
}


def fetch_dem(
    *,
    source_id: str,
    depth_lr_fp: str | Path,
    output_fp: str | Path | None = None,
    fetch_force_tiling: bool = False,
    logger=None,
) -> DemFetchResult:
    """Fetch a DEM for the given low-res depth tile using one registered source."""
    log = logger or logging.getLogger(__name__)
    source_key = str(source_id).strip().lower()
    assert source_key in _SOURCE_REGISTRY, f"unsupported DEM source_id='{source_id}'"
    log.debug(f"dispatching DEM fetch for source_id={source_key}")
    module_name, attr_name = _SOURCE_REGISTRY[source_key]
    # Import the selected backend only when the caller actually uses it.
    fetch_fn = getattr(import_module(module_name), attr_name)
    return fetch_fn(
        depth_lr_fp=depth_lr_fp,
        output_fp=output_fp,
        force_tiling=fetch_force_tiling,
        logger=log,
    )
