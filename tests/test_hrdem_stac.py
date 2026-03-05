"""Unit tests for HRDEM STAC helpers using synthetic rasters."""

from pathlib import Path

import numpy as np
import pytest

import floodsr.dem_sources.hrdem_stac
from conftest import _write_single_band_geotiff, logger


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "asset_crs",
    [
        pytest.param("EPSG:32633", id="asset_same_crs"),
        pytest.param("EPSG:3857", id="asset_crs_preserved_no_auto_reproject"),
    ],
)
def test_write_dem_from_asset_hrefs_synthetic_cases(
    tmp_path: Path,
    logger,
    asset_crs: str,
):
    """Synthetic assets should produce non-empty output while preserving source CRS."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import array_bounds, from_bounds, from_origin
    from rasterio.warp import transform_bounds

    # Create a low-res depth raster that defines query CRS and output bounds.
    depth_lr_fp = tmp_path / "depth_lr.tif"
    depth_arr = np.full((12, 12), 1.0, dtype=np.float32)
    depth_crs = "EPSG:32633"
    depth_transform = from_origin(500000.0, 4500000.0, 30.0, 30.0)
    _write_single_band_geotiff(depth_lr_fp, depth_arr, depth_transform, depth_crs, nodata=-9999.0)

    # Create one synthetic HRDEM asset, optionally in a different CRS.
    asset_fp = tmp_path / "asset_dem.tif"
    asset_arr = np.linspace(100.0, 140.0, 48 * 48, dtype=np.float32).reshape((48, 48))
    if asset_crs == depth_crs:
        asset_transform = from_bounds(*array_bounds(*depth_arr.shape, depth_transform), 48, 48)
    else:
        depth_bounds = array_bounds(*depth_arr.shape, depth_transform)
        asset_bounds = transform_bounds(depth_crs, asset_crs, *depth_bounds, densify_pts=21)
        asset_transform = from_bounds(*asset_bounds, 48, 48)
        
    _write_single_band_geotiff(asset_fp, asset_arr, asset_transform, asset_crs, nodata=-9999.0)

    # Build and read aligned output from synthetic asset hrefs.
    output_fp = tmp_path / "fetched_dem.tif"
    written_fp = floodsr.dem_sources.hrdem_stac.write_dem_from_asset_hrefs(
        depth_lr_fp=depth_lr_fp,
        asset_hrefs=[str(asset_fp)],
        output_fp=output_fp,
        logger=logger,
    )
    with rasterio.open(written_fp) as ds:
        fetched_dem = ds.read(1)
        fetched_crs = ds.crs

    assert fetched_dem.dtype == np.float32
    assert fetched_dem.size > 0
    assert fetched_crs == rasterio.crs.CRS.from_string(asset_crs)
    print(f"completed synthetic write_dem_from_asset_hrefs case for {asset_crs}")
