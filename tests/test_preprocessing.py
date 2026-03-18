"""Tests for preprocessing utilities."""

from pathlib import Path

import numpy as np
import pytest

from conftest import logger, synthetic_tohr_tiles
from floodsr.dem_sources.hrdem_mosaic import write_dem_from_asset_hrefs
from floodsr.preprocessing import write_platform_prepared_rasters, write_prepared_rasters


pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    "use_windowed",
    [
        pytest.param(False, id="prepared_simple"),
        pytest.param(True, id="prepared_windowed"),
    ],
)
def test_write_prepared_rasters_outputs_exist_and_are_float32(
    synthetic_tohr_tiles: dict,
    tmp_path,
    logger,
    use_windowed: bool,
) -> None:
    """Prepared outputs should exist on disk and keep float32 ToHR arrays."""
    rasterio = pytest.importorskip("rasterio")
    prepared = write_prepared_rasters(
        depth_lr_fp=synthetic_tohr_tiles["depth_lr_fp"],
        dem_hr_fp=synthetic_tohr_tiles["dem_fp"],
        scale=16,
        out_dir=tmp_path,
        logger=logger,
        use_windowed=use_windowed,
    )

    assert prepared["depth_lr_prepared_fp"].exists()
    assert prepared["dem_hr_prepared_fp"].exists()
    with rasterio.open(prepared["dem_hr_prepared_fp"]) as ds:
        dem_array = ds.read(1)
    assert dem_array.dtype == np.float32
    assert dem_array.size > 0


@pytest.mark.parametrize(
    "crs_policy, expected_crs",
    [
        pytest.param("use-lores", "EPSG:32633", id="crs_policy_use_lores"),
        pytest.param("use-dem", "EPSG:3857", id="crs_policy_use_dem"),
    ],
)
def test_write_prepared_rasters_honors_crs_policy_for_mismatch(
    tmp_path: Path,
    logger,
    crs_policy: str,
    expected_crs: str,
) -> None:
    """Mismatched CRS inputs should align to the policy-selected CRS."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import array_bounds, from_bounds, from_origin
    from rasterio.warp import transform_bounds

    depth_fp = tmp_path / "depth_mismatch.tif"
    dem_fp = tmp_path / "dem_mismatch.tif"
    depth_arr = np.full((16, 16), 1.0, dtype=np.float32)
    dem_arr = np.linspace(100.0, 120.0, 64 * 64, dtype=np.float32).reshape((64, 64))
    depth_crs = "EPSG:32633"
    dem_crs = "EPSG:3857"
    depth_transform = from_origin(500000.0, 4500000.0, 30.0, 30.0)
    depth_bounds = array_bounds(depth_arr.shape[0], depth_arr.shape[1], depth_transform)
    dem_bounds = transform_bounds(depth_crs, dem_crs, *depth_bounds, densify_pts=21)
    dem_transform = from_bounds(*dem_bounds, width=dem_arr.shape[1], height=dem_arr.shape[0])
    nodata = -9999.0

    depth_profile = {
        "driver": "GTiff",
        "height": int(depth_arr.shape[0]),
        "width": int(depth_arr.shape[1]),
        "count": 1,
        "dtype": "float32",
        "crs": depth_crs,
        "transform": depth_transform,
        "nodata": nodata,
    }
    dem_profile = {
        "driver": "GTiff",
        "height": int(dem_arr.shape[0]),
        "width": int(dem_arr.shape[1]),
        "count": 1,
        "dtype": "float32",
        "crs": dem_crs,
        "transform": dem_transform,
        "nodata": nodata,
    }
    with rasterio.open(depth_fp, "w", **depth_profile) as ds:
        ds.write(depth_arr, 1)
    with rasterio.open(dem_fp, "w", **dem_profile) as ds:
        ds.write(dem_arr, 1)

    prepared = write_prepared_rasters(
        depth_lr_fp=depth_fp,
        dem_hr_fp=dem_fp,
        scale=4,
        crs_policy=crs_policy,
        out_dir=tmp_path,
        logger=logger,
    )

    expected = rasterio.crs.CRS.from_string(expected_crs)
    with rasterio.open(prepared["depth_lr_prepared_fp"]) as ds_depth, rasterio.open(prepared["dem_hr_prepared_fp"]) as ds_dem:
        assert ds_depth.crs == expected
        assert ds_dem.crs == expected


def test_write_prepared_rasters_default_strict_rejects_crs_mismatch(
    tmp_path: Path,
    logger,
) -> None:
    """Default strict policy should fail when depth and DEM CRS differ."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    depth_fp = tmp_path / "depth_strict.tif"
    dem_fp = tmp_path / "dem_strict.tif"
    arr = np.full((8, 8), 1.0, dtype=np.float32)
    nodata = -9999.0

    with rasterio.open(
        depth_fp,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32633",
        transform=from_origin(500000.0, 4500000.0, 30.0, 30.0),
        nodata=nodata,
    ) as ds:
        ds.write(arr, 1)
    with rasterio.open(
        dem_fp,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(1660000.0, 6700000.0, 30.0, 30.0),
        nodata=nodata,
    ) as ds:
        ds.write(arr, 1)

    with pytest.raises(AssertionError, match="CRS mismatch under --crs-policy strict"):
        write_prepared_rasters(
            depth_lr_fp=depth_fp,
            dem_hr_fp=dem_fp,
            scale=4,
            out_dir=tmp_path,
            logger=logger,
        )


@pytest.mark.parametrize(
    "crs_policy, expected_crs",
    [
        pytest.param("use-lores", "EPSG:32633", id="platform_crs_policy_use_lores"),
        pytest.param("use-dem", "EPSG:3857", id="platform_crs_policy_use_dem"),
    ],
)
def test_write_platform_prepared_rasters_honors_crs_policy(
    tmp_path: Path,
    logger,
    crs_policy: str,
    expected_crs: str,
) -> None:
    """Platform preprocessing should harmonize outputs to the policy-selected CRS."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import array_bounds, from_bounds, from_origin
    from rasterio.warp import transform_bounds

    depth_fp = tmp_path / "platform_depth_mismatch.tif"
    dem_fp = tmp_path / "platform_dem_mismatch.tif"
    depth_arr = np.full((16, 16), 1.0, dtype=np.float32)
    dem_arr = np.linspace(100.0, 120.0, 64 * 64, dtype=np.float32).reshape((64, 64))
    depth_crs = "EPSG:32633"
    dem_crs = "EPSG:3857"
    depth_transform = from_origin(500000.0, 4500000.0, 30.0, 30.0)
    depth_bounds = array_bounds(depth_arr.shape[0], depth_arr.shape[1], depth_transform)
    dem_bounds = transform_bounds(depth_crs, dem_crs, *depth_bounds, densify_pts=21)
    dem_transform = from_bounds(*dem_bounds, width=dem_arr.shape[1], height=dem_arr.shape[0])
    nodata = -9999.0

    with rasterio.open(
        depth_fp,
        "w",
        driver="GTiff",
        height=depth_arr.shape[0],
        width=depth_arr.shape[1],
        count=1,
        dtype="float32",
        crs=depth_crs,
        transform=depth_transform,
        nodata=nodata,
    ) as ds:
        ds.write(depth_arr, 1)
    with rasterio.open(
        dem_fp,
        "w",
        driver="GTiff",
        height=dem_arr.shape[0],
        width=dem_arr.shape[1],
        count=1,
        dtype="float32",
        crs=dem_crs,
        transform=dem_transform,
        nodata=nodata,
    ) as ds:
        ds.write(dem_arr, 1)

    prepared = write_platform_prepared_rasters(
        depth_lr_fp=depth_fp,
        dem_hr_fp=dem_fp,
        out_dir=tmp_path,
        crs_policy=crs_policy,
        logger=logger,
    )

    expected = rasterio.crs.CRS.from_string(expected_crs)
    with rasterio.open(prepared["depth_lr_prepared_fp"]) as ds:
        depth_prepared = ds.read(1)
        prepared_crs = ds.crs
    assert prepared_crs == expected
    assert depth_prepared.size > 0


def test_write_dem_from_asset_hrefs_outputs_float32_non_empty(
    synthetic_tohr_tiles: dict,
    tmp_path: Path,
    logger,
) -> None:
    """Fetch-write helper should produce a readable float32 DEM raster."""
    rasterio = pytest.importorskip("rasterio")
    output_fp = tmp_path / "fetched_hrdem.tif"
    written_fp = write_dem_from_asset_hrefs(
        depth_lr_fp=synthetic_tohr_tiles["depth_lr_fp"],
        asset_hrefs=[str(synthetic_tohr_tiles["dem_fp"])],
        output_fp=output_fp,
        logger=logger,
    )
    with rasterio.open(written_fp) as ds:
        fetched_dem = ds.read(1)
    assert fetched_dem.dtype == np.float32
    assert fetched_dem.size > 0
