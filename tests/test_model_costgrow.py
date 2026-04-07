"""Model tests for CostGrow_Terrain: unit and integration."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
rasterio = pytest.importorskip("rasterio")

import floodsr.models.CostGrow_Terrain as costgrow_module
import floodsr.tohr
from floodsr.models.CostGrow_Terrain import (
    ModelWorker,
    _compute_cost_surface,
    _fill_nearest_unmasked,
    _filter_isolated,
)
from model_test_utils import assert_hard_only_windowed_selection, assert_result_raster_contract


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
):
    """Write a deterministic synthetic raster for windowed CostGrow plumbing tests."""
    del pcraster_module, depth_lr, depth_lr_profile, fine_profile
    del min_depth_threshold, dp_coarse_pixel_max, decay_frac, distance_fill_method, distance_fill_kwargs, show_progress
    with rasterio.open(dem_fine_fp) as ds:
        profile = ds.profile.copy()
        valid = ds.read_masks(1) > 0
    pred = np.where(valid, 0.25, float(out_nodata)).astype(np.float32, copy=False)
    profile.update(dtype="float32", count=1, nodata=float(out_nodata))
    if not bool(profile.get("tiled", False)):
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
    with rasterio.open(output_fp, "w", **profile) as dst:
        dst.write(pred, 1)
    return Path(output_fp), {"wet_anchors": 1, "wet_final": int(valid.sum())}


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
    assert result["preprocess"]["costgrow"]["windowed_contract"] == "transitional_disk_backed_global"


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
