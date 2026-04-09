"""Model tests for CostGrow_Terrain: unit and integration."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
rasterio = pytest.importorskip("rasterio")

import floodsr.models.CostGrow_Terrain as costgrow_module
import floodsr.tohr
from conftest import assert_hard_only_windowed_selection, assert_result_raster_contract
from floodsr.models.CostGrow_Terrain import (
    ModelWorker,
    _compute_cost_surface,
    _fill_nearest_unmasked,
    _filter_isolated,
)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.fast


def test_costgrow_cost_surface_penalizes_below_ground_cells():
    """Ensure the terrain penalty cost is 1 over wet cells and higher below ground."""
    wse = np.array([[5.0, 5.0], [2.0, 1.0]], dtype=np.float32)
    dem = np.array([[4.0, 5.5], [1.0, 3.0]], dtype=np.float32)
    valid = np.array([[True, True], [True, False]])
    cost, delta = _compute_cost_surface(wse, dem, valid)
    assert np.isclose(cost[0, 0], 1.0)
    assert np.isclose(cost[0, 1], 1.5)
    assert np.isclose(cost[1, 0], 1.0)
    assert np.isnan(cost[1, 1])
    assert np.isclose(delta[0, 1], -0.5)


def test_costgrow_filter_isolated_keeps_only_anchor_connected_region():
    """Ensure isolated grown regions are removed when disconnected from anchors."""
    source = np.array(
        [
            [True, True, False, False],
            [False, True, False, True],
            [False, False, False, True],
        ],
        dtype=bool,
    )
    anchor = np.array(
        [
            [True, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )
    filtered = _filter_isolated(source, anchor)
    assert filtered[0, 0]
    assert filtered[0, 1]
    assert filtered[1, 1]
    assert not filtered[1, 3]
    assert not filtered[2, 3]


def test_costgrow_fill_nearest_unmasked_returns_copy_when_array_has_no_mask():
    """Ensure fully wet coarse WSE inputs bypass the nearest-fill transform cleanly."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    masked = np.ma.MaskedArray(arr, mask=np.zeros_like(arr, dtype=bool))
    filled = _fill_nearest_unmasked(masked)
    assert np.array_equal(filled, arr)
    assert filled is not arr


def test_costgrow_worker_is_builtin_and_valid_without_artifact():
    """Ensure the built-in CostGrow worker does not require a model artifact."""
    worker = ModelWorker(model_fp=None)
    assert worker.requires_model_artifact is False
    assert worker.is_valid(None) is True


def test_costgrow_worker_resolves_windowed_path_only_for_hard_method():
    """Ensure windowed path requires window_method=='hard' AND sufficient raster size."""
    worker = ModelWorker(model_fp=None)
    assert_hard_only_windowed_selection(worker)


def test_costgrow_worker_builds_bounded_tile_contract():
    """Ensure windowed CostGrow advertises an explicit bounded-region tile contract."""
    worker = ModelWorker(model_fp=None)
    contract = worker._resolve_windowed_tile_contract(downscale=16, dp_coarse_pixel_max=10)
    assert contract["mode"] == "tile_halo"
    assert contract["core_tile_size_px"] == worker.windowed_core_tile_size_px
    assert contract["anchor_radius_px"] == 160
    assert contract["halo_px"] == 320
    assert contract["merge_rule"] == "hard_crop_core"
    assert contract["staged_state"] == "global_coarse_prefill_plus_tile_local_recompute"


@pytest.mark.parametrize(
    "dp_coarse_pixel_max",
    [
        pytest.param(None, id="missing_dp_coarse_pixel_max"),
        pytest.param(-1, id="negative_dp_coarse_pixel_max"),
    ],
)
def test_costgrow_worker_rejects_invalid_windowed_tile_contract_inputs(dp_coarse_pixel_max):
    """Ensure bounded tile contracts reject missing or negative growth-distance limits."""
    worker = ModelWorker(model_fp=None)
    with pytest.raises(AssertionError):
        worker._resolve_windowed_tile_contract(downscale=16, dp_coarse_pixel_max=dp_coarse_pixel_max)


@pytest.mark.parametrize(
    "core_window, halo_pixels, max_shape, expected_padded, expected_crop",
    [
        pytest.param(
            rasterio.windows.Window(col_off=40, row_off=30, width=20, height=10),
            5,
            (100, 100),
            rasterio.windows.Window(col_off=35, row_off=25, width=30, height=20),
            (slice(5, 15), slice(5, 25)),
            id="interior_tile",
        ),
        pytest.param(
            rasterio.windows.Window(col_off=0, row_off=0, width=20, height=10),
            5,
            (100, 100),
            rasterio.windows.Window(col_off=0, row_off=0, width=25, height=15),
            (slice(0, 10), slice(0, 20)),
            id="edge_clipped_tile",
        ),
    ],
)
def test_costgrow_windowed_geometry_helpers(core_window, halo_pixels, max_shape, expected_padded, expected_crop):
    """Ensure tile-halo padding and core-crop geometry stays aligned at interior and edge tiles."""
    padded_window = costgrow_module._expand_window(core_window, halo_pixels, max_shape)
    core_crop = costgrow_module._crop_from_padded_window(core_window, padded_window)
    assert padded_window == expected_padded
    assert core_crop == expected_crop


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


def _fake_costgrow_core(
    pcraster_module,
    depth_lr,
    depth_lr_profile,
    dem_lr,
    dem_lr_valid_mask,
    dem_fine,
    dem_fine_valid_mask,
    fine_profile,
    min_depth_threshold,
    dp_coarse_pixel_max,
    decay_frac,
    distance_fill_method,
    distance_fill_kwargs,
):
    """Return a deterministic synthetic depth field for integration plumbing tests."""
    del pcraster_module, depth_lr, depth_lr_profile, dem_lr, dem_lr_valid_mask, fine_profile
    del min_depth_threshold, dp_coarse_pixel_max, decay_frac, distance_fill_method, distance_fill_kwargs
    out = np.where(dem_fine_valid_mask, np.maximum(dem_fine * 0.0 + 0.25, 0.0), np.nan).astype(np.float32, copy=False)
    return out, {"wet_anchors": 1, "wet_final": int(np.isfinite(out).sum())}


def _fake_costgrow_core_windowed(
    pcraster_module,
    depth_lr,
    depth_lr_profile,
    dem_fine_fp,
    dem_fine_nodata,
    fine_profile,
    output_fp,
    out_nodata,
    min_depth_threshold,
    dp_coarse_pixel_max,
    decay_frac,
    distance_fill_method,
    distance_fill_kwargs,
    show_progress,
    tile_core_size_px,
    tile_halo_pixels,
):
    """Write a deterministic synthetic raster for windowed CostGrow plumbing tests."""
    del pcraster_module, depth_lr_profile, fine_profile
    del min_depth_threshold, decay_frac, distance_fill_method, distance_fill_kwargs, show_progress
    with rasterio.open(dem_fine_fp) as ds:
        profile = ds.profile.copy()
        valid = ds.read_masks(1) > 0
        downscale = int(round(int(ds.height) / int(depth_lr.shape[0])))
    pred = np.where(valid, 0.25, float(out_nodata)).astype(np.float32, copy=False)
    profile.update(dtype="float32", count=1, nodata=float(out_nodata))
    if not bool(profile.get("tiled", False)):
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
    with rasterio.open(output_fp, "w", **profile) as dst:
        dst.write(pred, 1)
    return Path(output_fp), {
        "wet_anchors": 1,
        "wet_final": int(valid.sum()),
        "tile_contract_mode": "tile_halo",
        "tile_core_size_px": int(tile_core_size_px),
        "tile_halo_px": int(tile_halo_pixels),
        "tile_anchor_radius_px": int(dp_coarse_pixel_max) * max(downscale, 1),
        "merge_rule": "hard_crop_core",
        "staged_state": "global_coarse_prefill_plus_tile_local_recompute",
    }


def _write_prepared_geotiff(fp: Path, array: np.ndarray, transform, crs: str) -> None:
    """Write a prepared-style single-band float32 GeoTIFF without nodata metadata."""
    profile = {
        "driver": "GTiff",
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "LZW",
    }
    with rasterio.open(fp, "w", **profile) as ds:
        ds.write(array.astype(np.float32, copy=False), 1)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_costgrow_tohr_runs_with_simple_platform_materialization(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_tohr_tiles: dict,
    logger,
):
    """Ensure CostGrow runs through `tohr()` and takes the simple path on small rasters."""
    monkeypatch.setattr(costgrow_module, "_check_pcraster", lambda: object())
    monkeypatch.setattr(costgrow_module, "_run_costgrow_core", _fake_costgrow_core)
    monkeypatch.setattr(floodsr.tohr, "resolve_model_worker_class", lambda _: costgrow_module.ModelWorker)

    result = floodsr.tohr.tohr(
        model_version="CostGrow_Terrain",
        model_fp=None,
        depth_lr_fp=synthetic_tohr_tiles["depth_lr_fp"],
        dem_hr_fp=synthetic_tohr_tiles["dem_fp"],
        output_fp=synthetic_tohr_tiles["output_fp"],
        window_method="hard",
        tile_overlap=0,
        logger=logger,
    )

    pred = assert_result_raster_contract(result, expected_shape=synthetic_tohr_tiles["hr_shape"])
    assert np.isfinite(pred).any()
    assert result["execution_path"] == "simple"
    assert result["platform_materialization"] == "simple"
    assert result["preprocess"]["window_method"] == "hard"
    assert result["preprocess"]["costgrow"]["windowed_contract"] == "whole_scene"


def test_costgrow_tohr_uses_windowed_path_for_large_hard_rasters(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_tohr_windowed_tiles: dict,
    logger,
):
    """Ensure CostGrow selects windowed execution for large rasters with window_method='hard'."""
    monkeypatch.setattr(costgrow_module, "_check_pcraster", lambda: object())
    monkeypatch.setattr(costgrow_module, "_run_costgrow_core_windowed", _fake_costgrow_core_windowed)
    monkeypatch.setattr(floodsr.tohr, "resolve_model_worker_class", lambda _: costgrow_module.ModelWorker)

    result = floodsr.tohr.tohr(
        model_version="CostGrow_Terrain",
        model_fp=None,
        depth_lr_fp=synthetic_tohr_windowed_tiles["depth_lr_fp"],
        dem_hr_fp=synthetic_tohr_windowed_tiles["dem_fp"],
        output_fp=synthetic_tohr_windowed_tiles["output_fp"],
        window_method="hard",
        tile_overlap=0,
        logger=logger,
    )

    pred = assert_result_raster_contract(result)
    assert np.isfinite(pred).any()
    assert result["execution_path"] == "windowed"
    assert result["platform_materialization"] == "windowed"
    assert result["preprocess"]["window_method"] == "hard"
    assert result["preprocess"]["costgrow"]["windowed_contract"] == "tile_halo"
    assert result["preprocess"]["costgrow"]["tile_contract_mode"] == "tile_halo"
    assert result["preprocess"]["costgrow"]["merge_rule"] == "hard_crop_core"
    assert result["preprocess"]["costgrow"]["staged_state"] == "global_coarse_prefill_plus_tile_local_recompute"


def test_costgrow_tohr_stays_simple_for_large_feather_rasters(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_tohr_windowed_tiles: dict,
    logger,
):
    """Ensure CostGrow stays on the simple path for large rasters when window_method='feather'."""
    monkeypatch.setattr(costgrow_module, "_check_pcraster", lambda: object())
    monkeypatch.setattr(costgrow_module, "_run_costgrow_core", _fake_costgrow_core)
    monkeypatch.setattr(floodsr.tohr, "resolve_model_worker_class", lambda _: costgrow_module.ModelWorker)

    result = floodsr.tohr.tohr(
        model_version="CostGrow_Terrain",
        model_fp=None,
        depth_lr_fp=synthetic_tohr_windowed_tiles["depth_lr_fp"],
        dem_hr_fp=synthetic_tohr_windowed_tiles["dem_fp"],
        output_fp=synthetic_tohr_windowed_tiles["output_fp"],
        window_method="feather",
        tile_overlap=0,
        logger=logger,
    )

    pred = assert_result_raster_contract(result)
    assert np.isfinite(pred).any()
    assert result["execution_path"] == "simple"
    assert result["preprocess"]["window_method"] == "feather"
    assert result["preprocess"]["costgrow"]["windowed_contract"] == "whole_scene"


def test_costgrow_windowed_tile_halo_executes_real_bounded_tiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    logger,
):
    """Ensure the real windowed CostGrow path runs bounded tiles and hard-crops core writes."""
    from rasterio.transform import from_origin

    monkeypatch.setattr(costgrow_module, "_check_pcraster", lambda: object())

    depth_lr = np.full((16, 16), 1.0, dtype=np.float32)
    dem_hr = np.zeros((256, 256), dtype=np.float32)
    crs = "EPSG:32633"
    depth_fp = tmp_path / "depth_lr_prepared.tif"
    dem_fp = tmp_path / "dem_hr_prepared.tif"
    output_fp = tmp_path / "pred_sr_windowed_real.tif"
    _write_prepared_geotiff(depth_fp, depth_lr, from_origin(0.0, 256.0, 16.0, 16.0), crs)
    _write_prepared_geotiff(dem_fp, dem_hr, from_origin(0.0, 256.0, 1.0, 1.0), crs)

    worker = ModelWorker(model_fp=None, logger=logger)
    worker.windowed_io_min_bytes = 0
    worker.windowed_core_tile_size_px = 64
    result = worker.run(
        depth_lr_fp=depth_fp,
        dem_hr_fp=dem_fp,
        output_fp=output_fp,
        window_method="hard",
        dp_coarse_pixel_max=1,
        show_progress=False,
    )

    pred = assert_result_raster_contract(result, expected_shape=dem_hr.shape)
    assert np.allclose(pred, 1.0)
    assert result["execution_path"] == "windowed"
    assert result["preprocess"]["window_method"] == "hard"
    assert result["preprocess"]["costgrow"]["windowed_contract"] == "tile_halo"
    assert result["preprocess"]["costgrow"]["tile_contract_mode"] == "tile_halo"
    assert result["preprocess"]["costgrow"]["tile_core_size_px"] == 64
    assert result["preprocess"]["costgrow"]["tile_anchor_radius_px"] == 16
    assert result["preprocess"]["costgrow"]["tile_halo_px"] == 32
    assert result["preprocess"]["costgrow"]["merge_rule"] == "hard_crop_core"
    assert result["preprocess"]["costgrow"]["staged_state"] == "global_coarse_prefill_plus_tile_local_recompute"
