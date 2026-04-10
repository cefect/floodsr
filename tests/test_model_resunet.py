"""Model tests for ResUNet_16x_DEM: unit, engine, and tiled inference."""

import pytest

np = pytest.importorskip("numpy")

import floodsr.models.ResUNet_16x_DEM as resunet_module
import floodsr.tohr
from conftest import assert_hard_only_windowed_selection, assert_result_raster_contract
from floodsr.engine import EngineORT


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_resunet_worker_resolves_windowed_path_only_for_hard_method():
    """Ensure windowed path requires window_method=='hard' AND sufficient raster size."""
    worker = object.__new__(resunet_module.ModelWorker)
    worker.windowed_io_min_bytes = resunet_module.ModelWorker.windowed_io_min_bytes
    assert_hard_only_windowed_selection(worker)


# ---------------------------------------------------------------------------
# Engine contract tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "repeat_run",
    [
        pytest.param(False, id="ort_contract_single_run"),
        pytest.param(True, id="ort_contract_repeat_run_is_deterministic"),
    ],
)
@pytest.mark.fast
def test_engine_ort_run_tile_contract(tohr_model_fp, ort_tile_inputs, logger, repeat_run: bool):
    """Ensure ORT predictions are float32, non-empty, and deterministic on repeat."""
    pytest.importorskip("onnxruntime")
    engine_instance = EngineORT(tohr_model_fp, logger=logger)
    run1 = engine_instance.run_tile(
        ort_tile_inputs["depth_lr"],
        ort_tile_inputs["dem_hr"],
        depth_lr_nodata=ort_tile_inputs["depth_lr_nodata"],
        dem_hr_nodata=ort_tile_inputs["dem_hr_nodata"],
        logger=logger,
    )
    assert run1["prediction_m"].dtype == np.float32
    assert run1["prediction_m"].size > 0

    if repeat_run:
        run2 = engine_instance.run_tile(
            ort_tile_inputs["depth_lr"],
            ort_tile_inputs["dem_hr"],
            depth_lr_nodata=ort_tile_inputs["depth_lr_nodata"],
            dem_hr_nodata=ort_tile_inputs["dem_hr_nodata"],
            logger=logger,
        )
        assert isinstance(run2["prediction_m"], np.ndarray)
        assert np.array_equal(run1["prediction_m"], run2["prediction_m"])


# ---------------------------------------------------------------------------
# Tiled inference integration tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
@pytest.mark.parametrize(
    "window_method, tile_overlap, expected_execution_path",
    [
        pytest.param("hard", 0, "simple", id="on_the_fly_synth_hard"),
        pytest.param("feather", 1, "simple", id="on_the_fly_synth_feather"),
    ],
)
def test_resunet_tohr_on_the_fly_synthetic_tiles(
    tohr_model_fp,
    default_model_version: str,
    synthetic_tohr_tiles: dict,
    window_method: str,
    tile_overlap: int,
    expected_execution_path: str,
    logger,
) -> None:
    """Run tiled ToHR on synthetic rasters for both window methods."""
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

    assert_result_raster_contract(result, expected_shape=synthetic_tohr_tiles["hr_shape"])
    assert result["execution_path"] == expected_execution_path


@pytest.mark.fast
def test_resunet_tohr_hard_windowed_tiles(
    tohr_model_fp,
    default_model_version: str,
    synthetic_tohr_windowed_tiles: dict,
    logger,
):
    """Run hard-window ToHR on a synthetic case that should trigger windowed IO."""
    result = floodsr.tohr.tohr(
        model_version=default_model_version,
        model_fp=tohr_model_fp,
        depth_lr_fp=synthetic_tohr_windowed_tiles["depth_lr_fp"],
        dem_hr_fp=synthetic_tohr_windowed_tiles["dem_fp"],
        output_fp=synthetic_tohr_windowed_tiles["output_fp"],
        window_method="hard",
        tile_overlap=0,
        logger=logger,
    )

    assert_result_raster_contract(result)
    assert result["execution_path"] == "windowed"
    assert result["output_fp"].endswith((".vrt" if resunet_module.gdal is not None else ".tif"))
