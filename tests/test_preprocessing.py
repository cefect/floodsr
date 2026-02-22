"""Tests for preprocessing utilities."""

from pathlib import Path

import numpy as np
import pytest

from floodsr.dem_sources.hrdem_stac import write_dem_from_asset_hrefs
from floodsr.preprocessing import write_prepared_rasters


pytestmark = pytest.mark.unit


def test_write_prepared_rasters_outputs_exist_and_are_float32(
    synthetic_inference_tiles: dict,
    tmp_path,
    logger,
) -> None:
    """Prepared outputs should exist on disk and keep float32 inference arrays."""
    rasterio = pytest.importorskip("rasterio")
    prepared = write_prepared_rasters(
        depth_lr_fp=synthetic_inference_tiles["depth_lr_fp"],
        dem_hr_fp=synthetic_inference_tiles["dem_fp"],
        scale=16,
        out_dir=tmp_path,
        logger=logger,
    )

    assert prepared["depth_lr_prepared_fp"].exists()
    assert prepared["dem_hr_prepared_fp"].exists()
    with rasterio.open(prepared["dem_hr_prepared_fp"]) as ds:
        dem_array = ds.read(1)
    assert dem_array.dtype == np.float32
    assert dem_array.size > 0


def test_write_dem_from_asset_hrefs_outputs_float32_non_empty(
    synthetic_inference_tiles: dict,
    tmp_path: Path,
    logger,
) -> None:
    """Fetch-write helper should produce a readable float32 DEM raster."""
    rasterio = pytest.importorskip("rasterio")
    output_fp = tmp_path / "fetched_hrdem.tif"
    written_fp = write_dem_from_asset_hrefs(
        depth_lr_fp=synthetic_inference_tiles["depth_lr_fp"],
        asset_hrefs=[str(synthetic_inference_tiles["dem_fp"])],
        output_fp=output_fp,
        logger=logger,
    )
    with rasterio.open(written_fp) as ds:
        fetched_dem = ds.read(1)
    assert fetched_dem.dtype == np.float32
    assert fetched_dem.size > 0
