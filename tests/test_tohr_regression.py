"""Tests for ToHR regression against committed case specs."""

import importlib.util
import json
import logging
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

def _write_derived_repeat_x_geotiff(src_fp: Path, dst_fp: Path, repeat_x: int) -> None:
    """Write a temporary raster by repeating the source array horizontally."""
    assert int(repeat_x) >= 1, f"repeat_x must be >= 1; got {repeat_x}"
    with rasterio.open(src_fp) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        arr_big = np.concatenate([arr] * int(repeat_x), axis=1)
        profile.update(width=int(arr_big.shape[1]))
        if not bool(profile.get("tiled", False)):
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
        with rasterio.open(dst_fp, "w", **profile) as dst:
            dst.write(arr_big, 1)


def _resolve_run_inputs(tile_dir: Path, case_inputs: dict, run_spec: dict, tmp_path: Path) -> dict[str, Path | bool]:
    """Resolve per-run input overrides and optional derived temporary rasters."""
    run_inputs = case_inputs.copy()
    run_inputs.update(run_spec.get("inputs", {}))
    input_derivation = run_spec.get("input_derivation")
    if not input_derivation:
        return {
            "depth_lr_fp": tile_dir / run_inputs["lowres_fp"],
            "dem_fp": run_inputs["dem_fp"] if run_inputs["dem_fp"] is False else tile_dir / run_inputs["dem_fp"],
            "truth_fp": run_inputs["truth_fp"] if run_inputs["truth_fp"] is False else tile_dir / run_inputs["truth_fp"],
        }

    mode = str(input_derivation.get("mode", "")).strip().lower()
    if mode != "repeat_x":
        raise AssertionError(f"unsupported input_derivation mode={mode!r}")
    repeat_x = int(input_derivation.get("repeat_x", 1))
    derived_d = {}
    for key, stem in (
        ("lowres_fp", "depth_lr"),
        ("dem_fp", "dem_hr"),
        ("truth_fp", "truth"),
    ):
        src_name = run_inputs[key]
        if src_name is False:
            derived_d[key] = False
            continue
        dst_fp = tmp_path / f"{run_spec['params']['model_version']}_{stem}_repeat_x{repeat_x}.tif"
        _write_derived_repeat_x_geotiff(tile_dir / src_name, dst_fp, repeat_x=repeat_x)
        derived_d[key] = dst_fp
    return {
        "depth_lr_fp": derived_d["lowres_fp"],
        "dem_fp": derived_d["dem_fp"],
        "truth_fp": derived_d["truth_fp"],
    }


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
    result_fp: Path | None = None,
    logger=None,
) -> dict | None:
    """Run CostGrow ToHR in a child interpreter and exit hard before native teardown."""
    write_result = result_fp is not None
    script = f"""
import os
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, {str(Path('.').resolve())!r})
import floodsr.tohr

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

result = floodsr.tohr.tohr(
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
if {write_result!r}:
    Path({str(result_fp)!r}).write_text(json.dumps(result), encoding='utf-8')
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
    log = logger or logging.getLogger(__name__)
    if result.stdout.strip():
        log.info(f"CostGrow subprocess stdout:\n    {result.stdout.rstrip()}")
    if result.stderr.strip():
        log.info(f"CostGrow subprocess stderr:\n    {result.stderr.rstrip()}")
    if result_fp is None:
        return None
    return json.loads(result_fp.read_text(encoding="utf-8"))

@pytest.mark.parametrize(
    "case_id,run_label",
    [
        pytest.param("2407_FHIMP_tile", "ResUNet_16x_DEM_default", id="fhimp_resunet"),
        pytest.param("2407_FHIMP_tile", "CostGrow_Terrain_default", id="fhimp_costgrow"),
        #pytest.param("fathom_n51w115", "ResUNet_16x_DEM_default", id="n51w115_resunet", marks=pytest.mark.local),
        pytest.param("rss_dudelange_A", "ResUNet_16x_DEM_default", id="dudelange_resunet"),
        #pytest.param("rss_dudelange_A", "CostGrow_Terrain_default", id="dudelange_costgrow", marks=pytest.mark.local), #ugly.. this is pluvial
        pytest.param("rss_mersch_A", "ResUNet_16x_DEM_default", id="mersch_resunet"),
        pytest.param("rss_mersch_A", "CostGrow_Terrain_default", id="mersch_costgrow"),
        pytest.param("rss_mersch_A", "CostGrow_Terrain_large_windowed", id="mersch_costgrow_large_windowed"),
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
    result = None
    input_fp_d = _resolve_run_inputs(tile_dir, case_spec["inputs"], run_spec, tmp_path)
    depth_lr_fp = input_fp_d["depth_lr_fp"]
    dem_fp = input_fp_d["dem_fp"]
    truth_fp = input_fp_d["truth_fp"]

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
        dem_hr_fp = dem_fp

    try:
        if model_version == "CostGrow_Terrain":
            result = _run_costgrow_tohr_in_subprocess(
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
                result_fp=tmp_path / f"{tile_case_d['case_name']}_{run_label}_result.json",
                logger=logger,
            )
        else:
            result = floodsr.tohr.tohr(
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
    expected_runtime = run_spec.get("runtime", {})
    if expected_runtime:
        assert result is not None, f"missing runtime result for case={tile_case_d['case_name']} run={run_label}"
        for key, expected_value in expected_runtime.items():
            if key == "windowed_contract":
                actual_value = result["preprocess"]["costgrow"]["windowed_contract"]
            elif key == "platform_materialization":
                actual_value = result["platform_materialization"]
            else:
                actual_value = result[key]
            assert actual_value == expected_value, (
                f"case={tile_case_d['case_name']} run={run_label} expected {key}={expected_value!r}, "
                f"got {actual_value!r}"
            )

    metrics = misc.eval.compute_depth_error_metrics_from_file(
        reference_fp=truth_fp,
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
