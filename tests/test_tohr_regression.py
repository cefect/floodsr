"""Tests for ToHR regression against committed case specs."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
np = pytest.importorskip("numpy")
rasterio = pytest.importorskip("rasterio")

import floodsr.dem_sources.catalog
import floodsr.tohr
import misc.eval
from conftest import logger, tile_case_d
from floodsr.model_registry import model_version_requires_artifact

pytestmark = pytest.mark.network


def _run_costgrow_tohr_in_subprocess(
    model_version: str,
    depth_lr_fp: Path,
    dem_hr_fp: Path,
    output_fp: Path,
    crs_policy: str,
    max_depth: float | None,
    dem_pct_clip: float | None,
    window_method: str,
    tile_overlap: int | None,
    tile_size: int | None,
) -> None:
    """Run CostGrow ToHR in a child interpreter and exit hard before native teardown."""
    script = f"""
import os
import sys
from pathlib import Path

sys.path.insert(0, {str(Path('.').resolve())!r})
import floodsr.tohr

floodsr.tohr.tohr(
    model_version={model_version!r},
    model_fp=None,
    depth_lr_fp=Path({str(depth_lr_fp)!r}),
    dem_hr_fp=Path({str(dem_hr_fp)!r}),
    output_fp=Path({str(output_fp)!r}),
    crs_policy={crs_policy!r},
    max_depth={max_depth!r},
    dem_pct_clip={dem_pct_clip!r},
    window_method={window_method!r},
    tile_overlap={tile_overlap!r},
    tile_size={tile_size!r},
    show_progress=False,
)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "CostGrow subprocess failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

@pytest.mark.parametrize(
    "case_id,run_label",
    [
        pytest.param("2407_FHIMP_tile", "ResUNet_16x_DEM_default", id="fhimp_resunet"),
        pytest.param("2407_FHIMP_tile", "CostGrow_Terrain_default", id="fhimp_costgrow"),
        #pytest.param("fathom_n51w115", "ResUNet_16x_DEM_default", id="n51w115_resunet", marks=pytest.mark.local),
        pytest.param("rss_dudelange_A", "ResUNet_16x_DEM_default", id="dudelange_resunet", marks=pytest.mark.local),
        pytest.param("rss_dudelange_A", "CostGrow_Terrain_default", id="dudelange_costgrow", marks=pytest.mark.local),
        pytest.param("rss_mersch_A", "ResUNet_16x_DEM_default", id="mersch_resunet", marks=pytest.mark.local),
        pytest.param("rss_mersch_A", "CostGrow_Terrain_default", id="mersch_costgrow", marks=pytest.mark.local),
    ],
)
@pytest.mark.local
def test_tohr_regression_matches_case_spec_metrics(
    tile_case_d: dict,
    run_label: str,
    tmp_path: Path,
    logger,
    request: pytest.FixtureRequest,
):
    """Validate ToHR metrics for all data-driven case specs via direct tohr invocation."""
    case_spec = tile_case_d["case_spec"]
    tile_dir = tile_case_d["tile_dir"]
    run_spec = case_spec["expected"][run_label]
    run_params = run_spec["params"].copy()
    model_version = run_params["model_version"]
    if model_version == "CostGrow_Terrain":
        if importlib.util.find_spec("pcraster") is None:
            pytest.skip("pcraster not detected in environment")
    else:
        pytest.importorskip("onnxruntime")
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
        if model_version == "CostGrow_Terrain":
            _run_costgrow_tohr_in_subprocess(
                model_version=model_version,
                depth_lr_fp=depth_lr_fp,
                dem_hr_fp=dem_hr_fp,
                output_fp=output_fp,
                crs_policy=run_params.get(
                    "crs_policy",
                    "use-dem" if case_spec["flags"]["in_hrdem"] and dem_fp is False else "strict",
                ),
                max_depth=run_params.get("max_depth"),
                dem_pct_clip=run_params.get("dem_pct_clip"),
                window_method=run_params.get("window_method", "hard"),
                tile_overlap=run_params.get("tile_overlap"),
                tile_size=run_params.get("tile_size"),
            )
        else:
            floodsr.tohr.tohr(
                model_version=model_version,
                model_fp=request.getfixturevalue("tohr_model_fp") if model_version_requires_artifact(model_version) else None,
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
    assert truth_fp is not False, f"missing truth_fp for case={tile_case_d['case_name']} run={run_label}"
    assert bool(case_spec["flags"].get("supports_regression_metrics", True)), (
        f"case={tile_case_d['case_name']} run={run_label} is not eligible for regression metrics"
    )

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




