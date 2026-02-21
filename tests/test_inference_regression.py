"""Tests for inference regression and synthetic tiling behavior."""

from pathlib import Path

import numpy as np
import pytest

from floodsr.inference import compute_depth_error_metrics, infer
from floodsr.preprocessing import replace_nodata_with_zero


_DATA_DRIVEN_CASES = [
    pytest.param("2407_FHIMP_tile", id="data_case_2407_fhimp_tile"),
    pytest.param("rss_mersch_A", id="data_case_rss_mersch_a"),
    pytest.param("rss_dudelange_A", id="data_case_rss_dudelange_a"),
]

_ON_THE_FLY_SYNTH_CASES = [
    pytest.param("hard", 0, id="on_the_fly_synth_hard"),
    pytest.param("feather", 1, id="on_the_fly_synth_feather"),
]


@pytest.mark.parametrize("tile_case", _DATA_DRIVEN_CASES, indirect=True)
def test_inference_regression_matches_case_spec_metrics(
    inference_model_fp: Path,
    tile_case: dict,
    tmp_path: Path,
    logger,
) -> None:
    """Validate inference metrics for all data-driven case specs."""
    pytest.importorskip("onnxruntime")
    rasterio = pytest.importorskip("rasterio")
    case_spec = tile_case["case_spec"]
    tile_dir = tile_case["tile_dir"]
    output_fp = tmp_path / f"{tile_case['case_name']}_pred_sr.tif"

    infer(
        model_fp=inference_model_fp,
        depth_lr_fp=tile_dir / case_spec["inputs"]["lowres_fp"],
        dem_hr_fp=tile_dir / case_spec["inputs"]["dem_fp"],
        output_fp=output_fp,
        logger=logger,
    )

    with rasterio.open(output_fp) as ds:
        pred = ds.read(1).astype(np.float32)
    with rasterio.open(tile_dir / case_spec["inputs"]["truth_fp"]) as ds:
        truth_raw = ds.read(1).astype(np.float32)
        truth_nodata = ds.nodata

    truth = replace_nodata_with_zero(truth_raw, truth_nodata)
    metrics = compute_depth_error_metrics(reference_depth_m=truth, estimate_depth_m=pred, max_depth=5.0)
    precision = int(case_spec["expected"].get("precision", 3))
    rounded_actual = {
        "mase_m": round(float(metrics["mase_m"]), precision),
        "rmse_m": round(float(metrics["rmse_m"]), precision),
        "ssim": round(float(metrics["ssim"]), precision),
    }
    rounded_expected = {
        "mase_m": round(float(case_spec["expected"]["metrics"]["mase_m"]), precision),
        "rmse_m": round(float(case_spec["expected"]["metrics"]["rmse_m"]), precision),
        "ssim": round(float(case_spec["expected"]["metrics"]["ssim"]), precision),
    }

    assert isinstance(case_spec["flags"]["in_hrdem"], bool)
    assert pred.dtype == np.float32
    assert pred.size > 0
    assert rounded_actual == rounded_expected


@pytest.mark.parametrize("window_method, tile_overlap", _ON_THE_FLY_SYNTH_CASES)
def test_infer_on_the_fly_synthetic_tiles(
    inference_model_fp: Path,
    synthetic_inference_tiles: dict,
    window_method: str,
    tile_overlap: int,
    logger,
) -> None:
    """Run tiled inference on on-the-fly synthetic rasters for both window methods."""
    pytest.importorskip("onnxruntime")
    rasterio = pytest.importorskip("rasterio")
    result = infer(
        model_fp=inference_model_fp,
        depth_lr_fp=synthetic_inference_tiles["depth_lr_fp"],
        dem_hr_fp=synthetic_inference_tiles["dem_fp"],
        output_fp=synthetic_inference_tiles["output_fp"],
        window_method=window_method,
        tile_overlap=tile_overlap,
        logger=logger,
    )

    with rasterio.open(result["output_fp"]) as ds:
        pred = ds.read(1)
    assert pred.shape == synthetic_inference_tiles["hr_shape"]
    assert pred.dtype == np.float32
    assert pred.size > 0
