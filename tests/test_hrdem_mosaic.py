"""Tests for HRDEM fetch helpers using synthetic and fixture-driven rasters."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

# Do not restore `import numpy as np` or `import rasterio` at module scope here.
# Use `pytest.importorskip(...)` instead so docs-environment discovery stays collection-safe; see docs/dev/adr/0006-tests.md.
np = pytest.importorskip("numpy", reason="HRDEM mosaic tests require numpy.")
rasterio = pytest.importorskip("rasterio", reason="HRDEM mosaic tests require rasterio.")

import floodsr.dem_sources.hrdem_mosaic
from conftest import _write_single_band_geotiff, logger, tile_case_d
from rasterio.transform import array_bounds, from_bounds, from_origin
from rasterio.windows import Window
from rasterio.warp import transform_bounds

# Match synthetic low-res raster origin to tests/data/2407_FHIMP_tile/lowres032.tif.
SYNTHETIC_DEPTH_ORIGIN_X = -1300733.0767616061
SYNTHETIC_DEPTH_ORIGIN_Y = 430719.82318666467

SYNTHETIC_REAL_FETCH_BASE_D = {
    "depth_crs": "EPSG:3979",
    "depth_res": 32.0,
    "force_tiling": True,
    "memory_limit_gib": 16.0,
}

SYNTHETIC_LOCAL_WRITE_BASE_D = {
    "depth_shape": (2, 2),
    "depth_res": 32.0,
    "depth_crs": "EPSG:32633",
    "asset_crs": "EPSG:32633",
    "asset_shape": (64, 64),
    "fetch_window_size": 32,
}


@pytest.fixture(scope="function")
def synthetic_lowres_builder(tmp_path: Path):
    """Build one synthetic low-res raster and return path/array/transform."""

    def _build(raster_name: str, depth_shape: tuple[int, int], depth_res: float, depth_crs: str):
        depth_lr_fp = tmp_path / f"{raster_name}.tif"
        depth_arr = np.full(depth_shape, 1.0, dtype=np.float32)
        depth_transform = from_origin(
            SYNTHETIC_DEPTH_ORIGIN_X,
            SYNTHETIC_DEPTH_ORIGIN_Y,
            float(depth_res),
            float(depth_res),
        )
        _write_single_band_geotiff(depth_lr_fp, depth_arr, depth_transform, depth_crs, nodata=-9999.0)
        return depth_lr_fp, depth_arr, depth_transform

    return _build


def _read_output_dem_with_basic_assertions(dem_fp: str | Path):
    """Read output DEM and assert dtype/non-empty basics used across tests."""
    with rasterio.open(dem_fp) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
    assert arr.dtype == np.float32
    assert arr.size > 0
    return arr, nodata


def _write_fetch_test_subset_depth_fp(tmp_path: Path, tile_case_d: dict) -> Path:
    """Write an optional low-res subset used to keep network fetch tests small."""
    case_spec = tile_case_d["case_spec"]
    tile_dir = tile_case_d["tile_dir"]
    depth_lr_fp = tile_dir / case_spec["inputs"]["lowres_fp"]
    subset_d = case_spec.get("fetch_test", {}).get("lowres_window")
    if subset_d is None:
        return depth_lr_fp

    subset_fp = tmp_path / f"{depth_lr_fp.stem}_fetch_subset.tif"
    subset_window = Window(
        col_off=int(subset_d["col_off"]),
        row_off=int(subset_d["row_off"]),
        width=int(subset_d["width"]),
        height=int(subset_d["height"]),
    )
    with rasterio.open(depth_lr_fp) as src_ds:
        subset_arr = src_ds.read(1, window=subset_window).astype(np.float32)
        subset_profile = src_ds.profile.copy()
        subset_profile.update(
            {
                "height": int(subset_window.height),
                "width": int(subset_window.width),
                "transform": src_ds.window_transform(subset_window),
            }
        )
    if not bool(subset_profile.get("tiled", False)):
        subset_profile.pop("blockxsize", None)
        subset_profile.pop("blockysize", None)
    with rasterio.open(subset_fp, "w", **subset_profile) as subset_ds:
        subset_ds.write(subset_arr, 1)
    return subset_fp








# ------------------
# ----- TESTSE -----
# ------------------

def test_build_fetch_tile_grid_gdf_and_selection_mask_writes_geojson(
    tmp_path: Path,
    logger,
):
    """Tile-grid helper should build a GeoDataFrame, support selection masking, and write GeoJSON."""
    gpd = pytest.importorskip("geopandas")
    tile_grid_gdf = floodsr.dem_sources.hrdem_mosaic._build_fetch_tile_grid_gdf(
        source_width=64,
        source_height=64,
        fetch_window_size=32,
        source_transform=from_origin(500000.0, 4000000.0, 1.0, 1.0),
        source_crs="EPSG:3979",
    )
    first_geom = tile_grid_gdf.iloc[0].geometry
    min_dim = min(first_geom.bounds[2] - first_geom.bounds[0], first_geom.bounds[3] - first_geom.bounds[1])
    shrink_dist = float(min_dim) * 0.1
    shrunk_geom = first_geom.buffer(-shrink_dist)
    if shrunk_geom.is_empty:
        shrunk_geom = first_geom
    project_extent_gdf = gpd.GeoDataFrame(
        {"feature_id": [1]},
        geometry=[shrunk_geom],
        crs=tile_grid_gdf.crs,
    )
    fetch_mask = floodsr.dem_sources.hrdem_mosaic._resolve_fetch_tile_selection_mask(tile_grid_gdf, project_extent_gdf)
    assert len(tile_grid_gdf) == 4
    assert bool(fetch_mask.any()) is True
    assert bool((~fetch_mask).any()) is True

    tile_grid_geojson_fp = tmp_path / "synthetic_fetch_tile_grid.geojson"
    tile_grid_geojson_fp.write_text(tile_grid_gdf.to_json(), encoding="utf-8")
    assert tile_grid_geojson_fp.exists()
    tile_grid_geojson_d = json.loads(tile_grid_geojson_fp.read_text(encoding="utf-8"))
    assert tile_grid_geojson_d["type"] == "FeatureCollection"
    assert len(tile_grid_geojson_d["features"]) == len(tile_grid_gdf)

    logger.info(f"Built tile grid GeoDataFrame with {len(tile_grid_gdf)} tiles, {fetch_mask.sum()} selected for fetch, saved to\n{tile_grid_geojson_fp}")






# -----------------
# ----- TESTS -----
# -----------------


@pytest.mark.fast
@pytest.mark.network
@pytest.mark.local
@pytest.mark.parametrize(
    "case_id",
    [
        pytest.param(
            "fathom_clip",
            id="fathom_clip",
            marks=pytest.mark.local,
        ),
        pytest.param("wri_aqueduct_yeg", id="wri_aqueduct_yeg"),
    ],
)
def test_download_hrdem_project_extent_for_data_case(tmp_path: Path, logger, tile_case_d: dict):
    """Project extent service should return at least one intersecting feature for the fixture tile."""
    depth_lr_fp = _write_fetch_test_subset_depth_fp(tmp_path, tile_case_d)
    depth_query = floodsr.dem_sources.hrdem_mosaic._resolve_depth_query_geometry(depth_lr_fp)
    clip_bounds_3979 = transform_bounds(
        depth_query["depth_crs"],
        "EPSG:3979",
        *depth_query["depth_bounds"],
        densify_pts=21,
    )
    feature_l = floodsr.dem_sources.hrdem_mosaic._download_hrdem_project_extent_features(
        clip_bounds_3979,
        raster_shape=(int(depth_query["depth_height"]), int(depth_query["depth_width"])),
        logger=logger,
    )
    assert isinstance(feature_l, list)
    assert len(feature_l) > 0
    feature_with_geometry_l = [feature_d for feature_d in feature_l if feature_d.get("geometry") is not None]
    assert feature_with_geometry_l, "project extent response had no geometries to write"
    geojson_fp = tmp_path / f"{tile_case_d['case_name']}_project_extent.geojson"
    feature_collection = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3979"}},
        "features": feature_with_geometry_l,
    }
    geojson_fp.write_text(json.dumps(feature_collection), encoding="utf-8")
    assert geojson_fp.exists()
    geojson_d = json.loads(geojson_fp.read_text(encoding="utf-8"))
    assert geojson_d["crs"]["properties"]["name"] == "EPSG:3979"

    logger.info(f"Downloaded {len(feature_l)} features, {len(feature_with_geometry_l)} with geometry, saved to\n{geojson_fp}")




@pytest.mark.network
@pytest.mark.parametrize(
    "domain_d",
    [
        pytest.param(
            {
                **SYNTHETIC_REAL_FETCH_BASE_D,
                "depth_shape": (2, 2),
                "fetch_window_size": 80,
            },
            id="real_hrdem_tiled_small_a",
        ),
        pytest.param(
            {
                **SYNTHETIC_REAL_FETCH_BASE_D,
                "depth_shape": (2, 3),
                "fetch_window_size": 80,
            },
            id="real_hrdem_tiled_small_b",
        ),
    ],
)
def test_fetch_hrdem_synthetic_cases(
    tmp_path: Path,
    logger,
    synthetic_lowres_builder,
    domain_d: dict,
):
    """Synthetic low-res rasters should fetch real HRDEM quickly with small tiled windows."""
    depth_lr_fp, _, _ = synthetic_lowres_builder(
        "depth_lr_synth",
        domain_d["depth_shape"],
        domain_d["depth_res"],
        domain_d["depth_crs"],
    )

    # Use real STAC/HRDEM path with forced small windows to keep runtime fast.
    output_fp = tmp_path / "fetched_dem_synth.tif"
    result = floodsr.dem_sources.hrdem_mosaic.main_fetch_hrdem_for_lowres_tile(
        depth_lr_fp=depth_lr_fp,
        output_fp=output_fp,
        logger=logger,
        use_cache=False,
        force_tiling=bool(domain_d["force_tiling"]),
        fetch_window_size=int(domain_d["fetch_window_size"]),
        memory_limit_gib=float(domain_d["memory_limit_gib"]),
    )

    _read_output_dem_with_basic_assertions(result.dem_fp)


@pytest.mark.fast
@pytest.mark.network
def test_public_hrdem_asset_without_boto3_is_suppressed_during_hrdem_fetch(
    tmp_path: Path,
    synthetic_lowres_builder,
    caplog,
):
    """HRDEM fetches should suppress rasterio DummySession info logs for public S3-backed assets.

    Observed with rasterio 1.5.0: public HRDEM asset hrefs contain
    ``amazonaws.com``, which causes rasterio.session to probe AWS-session
    handling. When boto3 is absent, rasterio logs an ``INFO`` fallback to
    ``DummySession`` even though the read still succeeds. This test keeps the
    baseline rasterio behavior visible, then verifies that our fetch wrapper
    suppresses only that non-actionable info message.
    """
    depth_lr_fp, _, _ = synthetic_lowres_builder(
        "depth_lr_session_probe",
        (2, 2),
        SYNTHETIC_REAL_FETCH_BASE_D["depth_res"],
        SYNTHETIC_REAL_FETCH_BASE_D["depth_crs"],
    )
    depth_query = floodsr.dem_sources.hrdem_mosaic._resolve_depth_query_geometry(depth_lr_fp)
    _, asset_hrefs, _ = floodsr.dem_sources.hrdem_mosaic._query_hrdem_assets(
        depth_query["bbox_4326"],
        stac_url=floodsr.dem_sources.hrdem_mosaic.STAC_URL,
        collection=floodsr.dem_sources.hrdem_mosaic.COLLECTION,
        asset_key=floodsr.dem_sources.hrdem_mosaic.DEFAULT_ASSET,
        stac_query_limit=5,
    )
    asset_href = asset_hrefs[0]

    # Public HRDEM assets are S3-backed HTTPS URLs, which is what triggers the
    # rasterio.session AWS-session detection path.
    assert "amazonaws.com" in asset_href
    with caplog.at_level("INFO", logger="rasterio.session"):
        session_cls = rasterio.session.Session.cls_from_path(asset_href)

    if importlib.util.find_spec("boto3") is None:
        assert session_cls is rasterio.session.DummySession
        assert "boto3 not available, falling back to a DummySession." in caplog.text
        caplog.clear()
        output_fp = tmp_path / "fetched_dem_session_probe.tif"
        with caplog.at_level("INFO", logger="rasterio.session"):
            result = floodsr.dem_sources.hrdem_mosaic.main_fetch_hrdem_for_lowres_tile(
                depth_lr_fp=depth_lr_fp,
                output_fp=output_fp,
                use_cache=False,
                force_tiling=True,
                fetch_window_size=32,
                memory_limit_gib=16.0,
                show_progress=False,
            )
        assert Path(result.dem_fp).exists() is True
        assert "boto3 not available, falling back to a DummySession." not in caplog.text
    else:
        assert session_cls is rasterio.session.AWSSession









@pytest.mark.fast
@pytest.mark.parametrize(
    "domain_d",
    [
        pytest.param(
            {
                **SYNTHETIC_LOCAL_WRITE_BASE_D,
                "asset_coverage_x_frac": 1.0,
                "expect_any_nodata": False,
            },
            id="write_full_coverage",
        ),
        pytest.param(
            {
                **SYNTHETIC_LOCAL_WRITE_BASE_D,
                "asset_coverage_x_frac": 0.5,
                "expect_any_nodata": True,
            },
            id="write_partial_coverage",
        ),
    ],
)
def test_write_dem_from_asset_hrefs_synthetic_cases(
    tmp_path: Path,
    logger,
    synthetic_lowres_builder,
    domain_d: dict,
):
    """Synthetic local assets should produce expected nodata coverage without monkeypatching."""
    depth_lr_fp, depth_arr, depth_transform = synthetic_lowres_builder(
        "depth_lr_local_asset",
        domain_d["depth_shape"],
        domain_d["depth_res"],
        domain_d["depth_crs"],
    )
    depth_bounds = array_bounds(*depth_arr.shape, depth_transform)

    # Build local synthetic asset with full or partial x-coverage over the query footprint.
    asset_fp = tmp_path / "asset_dem_local.tif"
    asset_arr = np.linspace(
        100.0,
        140.0,
        domain_d["asset_shape"][0] * domain_d["asset_shape"][1],
        dtype=np.float32,
    ).reshape(domain_d["asset_shape"])
    asset_bounds_full = (
        depth_bounds
        if domain_d["asset_crs"] == domain_d["depth_crs"]
        else transform_bounds(domain_d["depth_crs"], domain_d["asset_crs"], *depth_bounds, densify_pts=21)
    )
    asset_bounds = (
        asset_bounds_full[0],
        asset_bounds_full[1],
        asset_bounds_full[0] + ((asset_bounds_full[2] - asset_bounds_full[0]) * float(domain_d["asset_coverage_x_frac"])),
        asset_bounds_full[3],
    )
    asset_transform = from_bounds(*asset_bounds, domain_d["asset_shape"][1], domain_d["asset_shape"][0])
    _write_single_band_geotiff(asset_fp, asset_arr, asset_transform, domain_d["asset_crs"], nodata=-9999.0)

    output_fp = tmp_path / "fetched_dem_local_asset.vrt"
    dem_fp = floodsr.dem_sources.hrdem_mosaic.write_dem_from_asset_hrefs(
        depth_lr_fp=depth_lr_fp,
        asset_hrefs=[str(asset_fp)],
        output_fp=output_fp,
        logger=logger,
        fetch_window_size=int(domain_d["fetch_window_size"]),
    )

    arr, nodata = _read_output_dem_with_basic_assertions(dem_fp)
    if bool(domain_d["expect_any_nodata"]):
        assert np.any(np.isclose(arr, np.float32(nodata)))
    else:
        assert not np.any(np.isclose(arr, np.float32(nodata)))


@pytest.mark.fast
@pytest.mark.parametrize(
    "asset_coverage_x_frac",
    [
        pytest.param(1.0, id="non_windowed_full_coverage"),
        pytest.param(0.5, id="non_windowed_partial_coverage"),
    ],
)
def test_write_dem_from_asset_hrefs_non_windowed_outputs_float32_non_empty(
    tmp_path: Path,
    logger,
    synthetic_lowres_builder,
    asset_coverage_x_frac: float,
):
    """Synthetic local assets should exercise the non-windowed write path."""
    depth_lr_fp, depth_arr, depth_transform = synthetic_lowres_builder(
        "depth_lr_local_asset_non_windowed",
        SYNTHETIC_LOCAL_WRITE_BASE_D["depth_shape"],
        SYNTHETIC_LOCAL_WRITE_BASE_D["depth_res"],
        SYNTHETIC_LOCAL_WRITE_BASE_D["depth_crs"],
    )
    depth_bounds = array_bounds(*depth_arr.shape, depth_transform)
    asset_fp = tmp_path / "asset_dem_local_non_windowed.tif"
    asset_arr = np.linspace(
        100.0,
        140.0,
        SYNTHETIC_LOCAL_WRITE_BASE_D["asset_shape"][0] * SYNTHETIC_LOCAL_WRITE_BASE_D["asset_shape"][1],
        dtype=np.float32,
    ).reshape(SYNTHETIC_LOCAL_WRITE_BASE_D["asset_shape"])
    asset_bounds = (
        depth_bounds[0],
        depth_bounds[1],
        depth_bounds[0] + ((depth_bounds[2] - depth_bounds[0]) * float(asset_coverage_x_frac)),
        depth_bounds[3],
    )
    asset_transform = from_bounds(
        *asset_bounds,
        SYNTHETIC_LOCAL_WRITE_BASE_D["asset_shape"][1],
        SYNTHETIC_LOCAL_WRITE_BASE_D["asset_shape"][0],
    )
    _write_single_band_geotiff(
        asset_fp,
        asset_arr,
        asset_transform,
        SYNTHETIC_LOCAL_WRITE_BASE_D["asset_crs"],
        nodata=-9999.0,
    )

    output_fp = tmp_path / "fetched_dem_local_asset_non_windowed.tif"
    dem_fp = floodsr.dem_sources.hrdem_mosaic.write_dem_from_asset_hrefs(
        depth_lr_fp=depth_lr_fp,
        asset_hrefs=[str(asset_fp)],
        output_fp=output_fp,
        logger=logger,
        fetch_window_size=None,
    )

    _read_output_dem_with_basic_assertions(dem_fp)


@pytest.mark.fast
def test_write_dem_from_asset_hrefs_honors_explicit_cache_dir(
    tmp_path: Path,
    logger,
    synthetic_lowres_builder,
):
    """Explicit cache_dir should persist fetched DEM tiles outside the system temp directory."""
    depth_lr_fp, depth_arr, depth_transform = synthetic_lowres_builder(
        "depth_lr_cache",
        SYNTHETIC_LOCAL_WRITE_BASE_D["depth_shape"],
        SYNTHETIC_LOCAL_WRITE_BASE_D["depth_res"],
        SYNTHETIC_LOCAL_WRITE_BASE_D["depth_crs"],
    )
    depth_bounds = array_bounds(depth_arr.shape[0], depth_arr.shape[1], depth_transform)
    asset_fp = tmp_path / "asset_cache_source.tif"
    cache_dir = tmp_path / "explicit_hrdem_cache"
    output_fp = tmp_path / "fetched_dem_cached.tif"
    output_fp_second = tmp_path / "fetched_dem_cached_second.tif"

    # Build one local asset that fully covers the synthetic query extent.
    asset_arr = np.arange(
        1,
        (SYNTHETIC_LOCAL_WRITE_BASE_D["asset_shape"][0] * SYNTHETIC_LOCAL_WRITE_BASE_D["asset_shape"][1]) + 1,
        dtype=np.float32,
    ).reshape(SYNTHETIC_LOCAL_WRITE_BASE_D["asset_shape"])
    asset_transform = from_bounds(
        *depth_bounds,
        SYNTHETIC_LOCAL_WRITE_BASE_D["asset_shape"][1],
        SYNTHETIC_LOCAL_WRITE_BASE_D["asset_shape"][0],
    )
    _write_single_band_geotiff(
        asset_fp,
        asset_arr,
        asset_transform,
        SYNTHETIC_LOCAL_WRITE_BASE_D["asset_crs"],
        nodata=-9999.0,
    )

    # Run the same request twice so the second output can reuse the first cached tile.
    dem_fp = floodsr.dem_sources.hrdem_mosaic.write_dem_from_asset_hrefs(
        depth_lr_fp=depth_lr_fp,
        asset_hrefs=[str(asset_fp)],
        output_fp=output_fp,
        cache_dir=cache_dir,
        use_cache=True,
        logger=logger,
        fetch_window_size=None,
    )
    dem_fp_second = floodsr.dem_sources.hrdem_mosaic.write_dem_from_asset_hrefs(
        depth_lr_fp=depth_lr_fp,
        asset_hrefs=[str(asset_fp)],
        output_fp=output_fp_second,
        cache_dir=cache_dir,
        use_cache=True,
        logger=logger,
        fetch_window_size=None,
    )

    arr, _ = _read_output_dem_with_basic_assertions(dem_fp)
    assert Path(dem_fp_second).exists() is True
    assert len(list((cache_dir / "floodsr_hrdem_tile_cache").glob("*.tif"))) == 1

 


@pytest.mark.network
@pytest.mark.local
@pytest.mark.parametrize(
    "case_id",
    [
        pytest.param("fathom_clip",id="fathom_clip",),
        pytest.param("wri_aqueduct_yeg", id="wri_aqueduct_yeg"),
    ],
)
@pytest.mark.parametrize(
    "use_cache",
    [
        pytest.param(False, id="NOcache"),
    ],
)
@pytest.mark.parametrize(
    "fetch_kwargs",
    [
        pytest.param(
            {
                "force_tiling": True,
                "fetch_window_size": 2048,
                "use_project_extent_filter": False,
            },
            id="tiling_source_window",
        ),
    ],
)
def test_fetch_hrdem_data_case(
    tmp_path: Path,
    logger,
    tile_case_d: dict,
    use_cache: bool,
    fetch_kwargs: dict,
):
    """Non-synthetic case should produce non-empty output via main_fetch_hrdem_for_lowres_tile."""
    case_spec = tile_case_d["case_spec"]
    fetch_test_d = case_spec.get("fetch_test", {})
    depth_lr_fp = _write_fetch_test_subset_depth_fp(tmp_path, tile_case_d)
    assert depth_lr_fp.exists(), f"missing non-synthetic fixture: {depth_lr_fp}"
    output_fp = tmp_path / f"{depth_lr_fp.stem}_fetch_use_cache_{int(use_cache)}.tif"
    fetch_kwargs = {
        **fetch_kwargs,
        "fetch_window_size": int(fetch_test_d.get("fetch_window_size", fetch_kwargs["fetch_window_size"])),
    }

    # Fetch DEM directly in-process for faster and clearer failure surfaces.
    result = floodsr.dem_sources.hrdem_mosaic.main_fetch_hrdem_for_lowres_tile(
        depth_lr_fp=depth_lr_fp,
        output_fp=output_fp,
        logger=logger,
        use_cache=use_cache,
        **fetch_kwargs,
    )

    assert result.source_id == "hrdem"
    assert Path(result.dem_fp).exists() is True
    _read_output_dem_with_basic_assertions(result.dem_fp)
    tile_dir = Path(result.dem_fp).parent / f"{Path(result.dem_fp).stem}__fetch_tiles"
    tile_fp_l = list(tile_dir.glob("*.tif"))
    assert len(tile_fp_l) > 0
    assert len(tile_fp_l) <= int(fetch_test_d.get("expected_max_tiles", 4))
