"""Tests for ToHR regression and synthetic tiling behavior."""

from pathlib import Path

import numpy as np
import pytest

import floodsr.dem_sources.catalog
import floodsr.tohr
import misc.eval
from conftest import default_model_version, logger, synthetic_tohr_tiles, tile_case_d, tohr_model_fp
import rasterio

pytestmark = pytest.mark.network

@pytest.mark.parametrize(
    "case_id,run_label",
    [
        pytest.param("2407_FHIMP_tile", "ResUNet_16x_DEM_default", id="fhimp_default"),
        pytest.param("fathom_clip", "ResUNet_16x_DEM_default", id="clip_default", marks=pytest.mark.local),
        #pytest.param("fathom_n51w115", "ResUNet_16x_DEM_default", id="n51w115_default", marks=pytest.mark.local),
        pytest.param("rss_dudelange_A", "ResUNet_16x_DEM_default", id="dudelange_default", marks=pytest.mark.local),
        pytest.param("rss_mersch_A", "ResUNet_16x_DEM_default", id="mersch_default", marks=pytest.mark.local),
    ],
)
def test_tohr_regression_matches_case_spec_metrics(
    tohr_model_fp: Path,
    tile_case_d: dict,
    run_label: str,
    tmp_path: Path,
    logger,
):
    """Validate ToHR metrics for all data-driven case specs via direct tohr invocation."""
    pytest.importorskip("onnxruntime")
    case_spec = tile_case_d["case_spec"]
    tile_dir = tile_case_d["tile_dir"]
    run_spec = case_spec["expected"][run_label]
    run_params = run_spec["params"].copy()
    output_fp = tmp_path / f"{tile_case_d['case_name']}_{run_label}_pred_sr.tif"
    depth_lr_fp = tile_dir / case_spec["inputs"]["lowres_fp"]
    dem_fp = case_spec["inputs"]["dem_fp"]
    truth_fp = case_spec["inputs"]["truth_fp"]

    assert isinstance(case_spec["flags"]["in_hrdem"], bool)
    if dem_fp is False:
        assert case_spec["flags"]["in_hrdem"] is True
        dem_hr_fp = floodsr.dem_sources.catalog.fetch_dem(
            source_id="hrdem",
            depth_lr_fp=depth_lr_fp,
            output_fp=tmp_path / f"{tile_case_d['case_name']}_{run_label}_fetch_dem.tif",
            logger=logger,
        ).dem_fp
    else:
        dem_hr_fp = tile_dir / dem_fp

    try:
        floodsr.tohr.tohr(
            model_version=run_params["model_version"],
            model_fp=tohr_model_fp,
            depth_lr_fp=depth_lr_fp,
            dem_hr_fp=dem_hr_fp,
            output_fp=output_fp,
            crs_policy=run_params.get(
                "crs_policy",
                "use-dem" if case_spec["flags"]["in_hrdem"] and dem_fp is False else "strict",
            ),
            max_depth=run_params.get("max_depth"),
            dem_pct_clip=run_params.get("dem_pct_clip"),
            window_method=run_params.get("window_method", "feather"),
            tile_overlap=run_params.get("tile_overlap"),
            tile_size=run_params.get("tile_size"),
            logger=logger,
        )
    except Exception as err:
        pytest.fail(f"tohr failed for case={tile_case_d['case_name']} run={run_label}; error={err}")

    with rasterio.open(output_fp) as ds:
        pred = ds.read(1)

    assert pred.dtype == np.float32
    assert pred.size > 0
    if not bool(case_spec["flags"].get("supports_regression_metrics", True)):
        pytest.xfail(
            f"case={tile_case_d['case_name']} run={run_label} lacks truth-backed regression metrics required by ADR-0006"
        )
    assert truth_fp is not False, f"missing truth_fp for case={tile_case_d['case_name']} run={run_label}"

    metrics = misc.eval.compute_depth_error_metrics_from_file(
        reference_fp=tile_dir / truth_fp,
        estimate_fp=output_fp,
        max_depth=5.0,
    )
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
    assert rounded_actual == rounded_expected


@pytest.mark.parametrize(
    "window_method, tile_overlap",
    [
        pytest.param("hard", 0, id="on_the_fly_synth_hard"),
        pytest.param("feather", 1, id="on_the_fly_synth_feather"),
    ],
)
def test_tohr_on_the_fly_synthetic_tiles(
    tohr_model_fp: Path,
    default_model_version: str,
    synthetic_tohr_tiles: dict,
    window_method: str,
    tile_overlap: int,
    logger,
) -> None:
    """Run tiled ToHR on on-the-fly synthetic rasters for both window methods."""
    # `tohr_model_fp` is provided by tests/conftest.py::tohr_model_fp.
    # Execute end-to-end inference for the selected synthetic tiling setup.
    result = floodsr.tohr.tohr(
        model_version=default_model_version,
        model_fp=tohr_model_fp,
        depth_lr_fp=synthetic_tohr_tiles["depth_lr_fp"],
        dem_hr_fp=synthetic_tohr_tiles["dem_fp"],
        output_fp=synthetic_tohr_tiles["output_fp"],
        window_method=window_method,
        tile_overlap=tile_overlap,
        logger=logger,
    )

    # Validate basic output contract for the synthetic test fixture.
    with rasterio.open(result["output_fp"]) as ds:
        pred = ds.read(1)
    assert pred.shape == synthetic_tohr_tiles["hr_shape"]
    assert pred.dtype == np.float32
    assert pred.size > 0
