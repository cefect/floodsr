"""Integration tests for CostGrow ToHR plumbing."""

from pathlib import Path

import pytest
np = pytest.importorskip("numpy")
rasterio = pytest.importorskip("rasterio")

import floodsr.models.CostGrow_Terrain as costgrow_module
import floodsr.tohr


pytestmark = pytest.mark.fast


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


def test_costgrow_tohr_runs_with_simple_platform_materialization(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_tohr_tiles: dict,
    logger,
):
    """Ensure CostGrow runs through `tohr()` and keeps the simple materialization path on small rasters."""
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

    with rasterio.open(result["output_fp"]) as ds:
        pred = ds.read(1)
    assert pred.shape == synthetic_tohr_tiles["hr_shape"]
    assert pred.dtype == np.float32
    assert np.isfinite(pred).any()
    assert result["execution_path"] == "simple"
    assert result["platform_materialization"] == "simple"


def test_costgrow_tohr_uses_windowed_platform_materialization_for_large_rasters(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_tohr_windowed_tiles: dict,
    logger,
):
    """Ensure CostGrow follows the same windowed platform-preparation trigger as ResUNet on large rasters."""
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

    assert Path(result["output_fp"]).exists()
    assert result["execution_path"] == "windowed"
    assert result["platform_materialization"] == "windowed"
