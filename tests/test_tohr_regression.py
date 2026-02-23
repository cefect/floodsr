"""Tests for ToHR regression and synthetic tiling behavior."""

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import TEST_TILE_CASES
from floodsr.cli import main
from floodsr.preprocessing import replace_nodata_with_zero
from floodsr.tohr import tohr
from misc.eval import compute_depth_error_metrics


pytestmark = pytest.mark.e2e

_DATA_DRIVEN_CASES = [pytest.param(case_name, id=f"data_case_{case_name.lower()}") for case_name in TEST_TILE_CASES]

_ON_THE_FLY_SYNTH_CASES = [
    pytest.param("hard", 0, id="on_the_fly_synth_hard"),
    pytest.param("feather", 1, id="on_the_fly_synth_feather"),
]


@pytest.mark.parametrize("tile_case", _DATA_DRIVEN_CASES, indirect=True)
def test_tohr_regression_matches_case_spec_metrics(
    tohr_model_fp: Path,
    tile_case: dict,
    tmp_path: Path,
) -> None:
    """Validate ToHR metrics for all data-driven case specs via machine-interface JSON CLI."""
    pytest.importorskip("onnxruntime")
    rasterio = pytest.importorskip("rasterio")
    case_spec = tile_case["case_spec"]
    tile_dir = tile_case["tile_dir"]
    with rasterio.open(tile_dir / case_spec["inputs"]["truth_fp"]) as ds:
        truth_raw = ds.read(1).astype(np.float32)
        truth_nodata = ds.nodata
    truth = replace_nodata_with_zero(truth_raw, truth_nodata)
    for run_label, run_spec in case_spec["expected"].items():
        # Build machine-interface payload directly from case spec params.
        output_fp = tmp_path / f"{tile_case['case_name']}_{run_label}_pred_sr.tif"
        machine_payload = {
            "in_fp": str(tile_dir / case_spec["inputs"]["lowres_fp"]),
            "dem": str(tile_dir / case_spec["inputs"]["dem_fp"]),
            "out": str(output_fp),
            "model_path": str(tohr_model_fp),
        }
        machine_payload.update(run_spec["params"])
        machine_json_fp = tmp_path / f"{tile_case['case_name']}_{run_label}_machine.json"
        machine_json_fp.write_text(json.dumps(machine_payload), encoding="utf-8")

        exit_code = main(["tohr", "--machine-json", str(machine_json_fp)])
        with rasterio.open(output_fp) as ds:
            pred = ds.read(1).astype(np.float32)
        metrics = compute_depth_error_metrics(reference_depth_m=truth, estimate_depth_m=pred, max_depth=5.0)
        precision = int(run_spec["metrics"].get("precision", 3))
        rounded_actual = {
            "mase_m": round(float(metrics["mase_m"]), precision),
            "rmse_m": round(float(metrics["rmse_m"]), precision),
            "ssim": round(float(metrics["ssim"]), precision),
        }
        rounded_expected = {
            "mase_m": round(float(run_spec["metrics"]["mase_m"]), precision),
            "rmse_m": round(float(run_spec["metrics"]["rmse_m"]), precision),
            "ssim": round(float(run_spec["metrics"]["ssim"]), precision),
        }

        assert isinstance(case_spec["flags"]["in_hrdem"], bool)
        assert isinstance(exit_code, int)
        assert exit_code == 0
        assert pred.dtype == np.float32
        assert pred.size > 0
        assert rounded_actual == rounded_expected


@pytest.mark.parametrize("window_method, tile_overlap", _ON_THE_FLY_SYNTH_CASES)
def test_tohr_on_the_fly_synthetic_tiles(
    tohr_model_fp: Path,
    default_model_version: str,
    synthetic_tohr_tiles: dict,
    window_method: str,
    tile_overlap: int,
    logger,
) -> None:
    """Run tiled ToHR on on-the-fly synthetic rasters for both window methods."""
    pytest.importorskip("onnxruntime")
    rasterio = pytest.importorskip("rasterio")
    result = tohr(
        model_version=default_model_version,
        model_fp=tohr_model_fp,
        depth_lr_fp=synthetic_tohr_tiles["depth_lr_fp"],
        dem_hr_fp=synthetic_tohr_tiles["dem_fp"],
        output_fp=synthetic_tohr_tiles["output_fp"],
        window_method=window_method,
        tile_overlap=tile_overlap,
        logger=logger,
    )

    with rasterio.open(result["output_fp"]) as ds:
        pred = ds.read(1)
    assert pred.shape == synthetic_tohr_tiles["hr_shape"]
    assert pred.dtype == np.float32
    assert pred.size > 0
