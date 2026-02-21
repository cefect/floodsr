"""Tests for inference utilities and raster execution path."""

from pathlib import Path

import numpy as np
import pytest
import rasterio

from floodsr.inference import compute_depth_error_metrics, infer
from floodsr.preprocessing import replace_nodata_with_zero
from tests.conftest import fhimp_tile_case, inference_model_fp, synthetic_inference_tiles, tile_case
 

 
_ARTIFACT_TILE_CASES = [
    pytest.param("2407_FHIMP_tile", id="2407_FHIMP_tile"),
    pytest.param("rss_mersch_A", id="rss_mersch_A"),
]

 


@pytest.mark.parametrize("tile_case", _ARTIFACT_TILE_CASES, indirect=True)
def test_inference_metrics_match_artifact_at_3dp(
    inference_model_fp: Path, tile_case: dict, tmp_path: Path,
    logger):
    """Validate end-to-end metrics for the known model tile artifact."""
    
    artifact = tile_case["artifact"]
    tile_dir = tile_case["tile_dir"]
    output_fp = tmp_path / "pred_sr.tif"

    infer(
        model_fp=inference_model_fp,
        depth_lr_fp=tile_dir / artifact["lowres_fp"],
        dem_hr_fp=tile_dir / artifact["dem_fp"],
        output_fp=output_fp,
        logger=logger,
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
    assert rounded_actual == rounded_expected


 


def test_infer_tiling_hard_for_synthetic_mismatched_ratio(
    inference_model_fp: Path,
    synthetic_inference_tiles: dict,
    logger,
):
    """Run hard-stitched tiling on intentionally mismatched raster ratio."""
 

    result = infer(
        model_fp=inference_model_fp,
        depth_lr_fp=synthetic_inference_tiles["depth_lr_fp"],
        dem_hr_fp=synthetic_inference_tiles["dem_fp"],
        output_fp=synthetic_inference_tiles["output_fp"],
        window_method="hard",
        tile_overlap=0,
        logger=logger,
    )

    with rasterio.open(result["output_fp"]) as ds:
        pred = ds.read(1)
    assert pred.shape == synthetic_inference_tiles["hr_shape"]
    assert pred.dtype == np.float32
    assert pred.size > 0


 
