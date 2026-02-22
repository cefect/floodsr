"""Registry and dispatch for DEM source backends."""

import logging
from pathlib import Path

from floodsr.dem_sources.base import DemFetchResult
from floodsr.dem_sources.hrdem_stac import fetch_hrdem_for_lowres_tile


_SOURCE_REGISTRY = {
    "hrdem": fetch_hrdem_for_lowres_tile,
}


def fetch_dem(
    *,
    source_id: str,
    depth_lr_fp: str | Path,
    output_fp: str | Path | None = None,
    logger=None,
) -> DemFetchResult:
    """Fetch a DEM for the given low-res depth tile using one registered source."""
    log = logger or logging.getLogger(__name__)
    source_key = str(source_id).strip().lower()
    assert source_key in _SOURCE_REGISTRY, f"unsupported DEM source_id='{source_id}'"
    log.debug(f"dispatching DEM fetch for source_id={source_key}")
    fetch_fn = _SOURCE_REGISTRY[source_key]
    return fetch_fn(
        depth_lr_fp=depth_lr_fp,
        output_fp=output_fp,
        logger=log,
    )
