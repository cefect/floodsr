"""Tests for inference utilities and GeoTIFF execution path."""

import json
from pathlib import Path

import numpy as np
import pytest

from floodsr.inference import compute_depth_error_metrics, infer_geotiff, replace_nodata_with_zero
from tests.conftest import inference_model_fp, model_tile_case, tile_case


def test_infer_geotiff_writes_prediction_raster(inference_model_fp, model_tile_case, tmp_path):
    """Ensure infer_geotiff writes a non-empty float32 prediction raster."""
    rasterio = pytest.importorskip("rasterio")
    artifact = model_tile_case["artifact"]
    tile_dir = model_tile_case["tile_dir"]
    output_fp = tmp_path / "pred_sr.tif"
    result = infer_geotiff(
        model_fp=inference_model_fp,
        depth_lr_fp=tile_dir / artifact["lowres_fp"],
        dem_hr_fp=tile_dir / artifact["dem_fp"],
        output_fp=output_fp,
    )
    with rasterio.open(Path(result["output_fp"])) as ds:
        pred = ds.read(1)
    assert pred.dtype == np.float32
    assert pred.size > 0


def test_inference_metrics_match_artifact_at_3dp(inference_model_fp, model_tile_case, tmp_path):
    """Validate model-only tile metrics against JSON artifact at configured precision."""
    rasterio = pytest.importorskip("rasterio")
    artifact = json.loads(Path(model_tile_case["metrics_fp"]).read_text(encoding="utf-8"))
    tile_dir = model_tile_case["tile_dir"]
    output_fp = tmp_path / "pred_sr.tif"
    infer_geotiff(
        model_fp=inference_model_fp,
        depth_lr_fp=tile_dir / artifact["lowres_fp"],
        dem_hr_fp=tile_dir / artifact["dem_fp"],
        output_fp=output_fp,
    )
    with rasterio.open(output_fp) as ds:
        pred = ds.read(1).astype(np.float32)
    with rasterio.open(tile_dir / artifact["truth_fp"]) as ds:
        truth_raw = ds.read(1).astype(np.float32)
        truth_nodata = ds.nodata
    truth = replace_nodata_with_zero(truth_raw, truth_nodata)
    metrics = compute_depth_error_metrics(reference_depth_m=truth, estimate_depth_m=pred, max_depth=5.0)
    precision = int(artifact.get("precision", 3))
    rounded_actual = {
        "mase_m": round(float(metrics["mase_m"]), precision),
        "rmse_m": round(float(metrics["rmse_m"]), precision),
        "ssim": round(float(metrics["ssim"]), precision),
    }
    rounded_expected = {
        "mase_m": round(float(artifact["metrics"]["mase_m"]), precision),
        "rmse_m": round(float(artifact["metrics"]["rmse_m"]), precision),
        "ssim": round(float(artifact["metrics"]["ssim"]), precision),
    }
    assert isinstance(metrics, dict)
    assert rounded_actual == rounded_expected


def test_infer_geotiff_defaults_output_driver_to_gtiff(inference_model_fp, model_tile_case, tmp_path):
    """Ensure output defaults to GTiff driver when extension does not map to a driver."""
    rasterio = pytest.importorskip("rasterio")
    artifact = model_tile_case["artifact"]
    tile_dir = model_tile_case["tile_dir"]
    output_fp = tmp_path / "pred_sr"
    result = infer_geotiff(
        model_fp=inference_model_fp,
        depth_lr_fp=tile_dir / artifact["lowres_fp"],
        dem_hr_fp=tile_dir / artifact["dem_fp"],
        output_fp=output_fp,
    )
    with rasterio.open(Path(result["output_fp"])) as ds:
        driver = ds.driver
    assert isinstance(driver, str)
    assert driver == "GTiff"


def test_tile_case_artifact_references_existing_rasters(tile_case):
    """Ensure each tile metrics artifact points to existing input/output rasters."""
    artifact = tile_case["artifact"]
    tile_dir = tile_case["tile_dir"]
    for key in ("lowres_fp", "dem_fp", "truth_fp"):
        raster_fp = tile_dir / artifact[key]
        assert raster_fp.exists()
