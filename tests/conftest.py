"""Pytest fixtures for FloodSR tests."""

import json
import logging
import pathlib

import numpy as np
import pytest


TEST_TILE_CASES = ("2407_FHIMP_tile", "rss_mersch_A", "rss_dudelange_A")


def _read_tile_case(case_name: str) -> dict:
    """Load tile case metadata from tests/data for a named fixture."""
    tile_dir = pathlib.Path("tests/data") / case_name
    metrics_fp = tile_dir / "metrics.json"
    assert tile_dir.exists(), f"missing tile directory: {tile_dir}"
    assert metrics_fp.exists(), f"missing metrics artifact: {metrics_fp}"
    return {
        "case_name": case_name,
        "tile_dir": tile_dir,
        "metrics_fp": metrics_fp,
        "artifact": json.loads(metrics_fp.read_text(encoding="utf-8")),
    }


def _write_single_band_geotiff(fp: pathlib.Path, array: np.ndarray, transform, crs: str, nodata: float = -9999.0) -> None:
    """Write a one-band float32 GeoTIFF with deterministic defaults."""
    import rasterio

    fp.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": float(nodata),
        "compress": "LZW",
    }
    with rasterio.open(fp, "w", **profile) as ds:
        ds.write(array.astype(np.float32), 1)


#===============================================================================
# pytest custom config------------
#===============================================================================


def pytest_runtest_teardown(item, nextitem):
    """Custom teardown message."""
    test_name = item.name
    print(f"\n{'='*20} Test completed: {test_name} {'='*20}\n\n\n")


def pytest_report_header(config):
    """Show pytest invocation arguments in the test header."""
    return f"pytest arguments: {' '.join(config.invocation_params.args)}"


# -------------------
# ----- Fixtures -----
# -------------------
@pytest.fixture(scope='session')
def logger():
    """Simple logger fixture for the function under test."""
    log = logging.getLogger("pytest")
    log.setLevel(logging.DEBUG)
    # keep handlers minimal to avoid duplicate logs across runs
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


@pytest.fixture(scope="session")
def tile_case_catalog():
    """Return metadata for all explicitly tracked tile fixtures."""
    return {case_name: _read_tile_case(case_name) for case_name in TEST_TILE_CASES}


@pytest.fixture
def tile_case(request, tile_case_catalog):
    """Return a tile case by explicit name parameter."""
    case_name = request.param
    assert case_name in tile_case_catalog, f"missing tile case in catalog: {case_name}"
    return tile_case_catalog[case_name]


@pytest.fixture(scope="session")
def fhimp_tile_case(tile_case_catalog):
    """Return the default FHIMP chip fixture."""
    return tile_case_catalog["2407_FHIMP_tile"]


@pytest.fixture(scope="session")
def rss_mersch_a_case(tile_case_catalog):
    """Return the RSS Mersch A chip fixture."""
    return tile_case_catalog["rss_mersch_A"]


@pytest.fixture(scope="session")
def ort_tile_inputs():
    """Create synthetic arrays that match a single model tile size."""
    return {
        "depth_lr": np.full((32, 32), 1.5, dtype=np.float32),
        "dem_hr": np.linspace(500.0, 1000.0, 512 * 512, dtype=np.float32).reshape((512, 512)),
        "depth_lr_nodata": -9999.0,
        "dem_hr_nodata": -9999.0,
    }


@pytest.fixture(scope="session")
def synthetic_inference_tiles(tmp_path_factory):
    """Create temporary raster inputs for on-the-fly inference coverage tests."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    root = tmp_path_factory.mktemp("inference_tiles")
    lr_shape = (64, 64)
    hr_shape = (960, 960)
    crs = "EPSG:32633"
    hr_resolution = 2.0
    lr_resolution = 30.0
    x0, y0 = 500000.0, 4000000.0

    depth_lr = np.full(lr_shape, 1.0, dtype=np.float32)
    dem = np.tile(np.linspace(500.0, 1000.0, hr_shape[1], dtype=np.float32), (hr_shape[0], 1))

    depth_lr_fp = root / "depth_lr.tif"
    dem_fp = root / "dem.tif"

    _write_single_band_geotiff(
        depth_lr_fp,
        depth_lr,
        from_origin(x0, y0 + lr_shape[0] * lr_resolution, lr_resolution, lr_resolution),
        crs,
    )
    _write_single_band_geotiff(
        dem_fp,
        dem,
        from_origin(x0, y0 + hr_shape[0] * hr_resolution, hr_resolution, hr_resolution),
        crs,
    )

    return {
        "depth_lr_fp": depth_lr_fp,
        "dem_fp": dem_fp,
        "hr_shape": hr_shape,
        "output_fp": root / "pred_sr.tif",
    }


@pytest.fixture(scope="function")
def inference_model_fp(tmp_path):
    """Resolve local model path used by engine and CLI inference tests."""
    model_version = "4690176_0_1770580046_train_base_16"
    local_fp = pathlib.Path("_inputs") / model_version / "model_infer.onnx"
    if local_fp.exists():
        return local_fp.resolve()

    from floodsr.model_registry import fetch_model

    try:
        return fetch_model(model_version, cache_dir=tmp_path / "cache")
    except Exception as exc:  # pragma: no cover - exercised by test skip behavior
        pytest.skip(f"unable to resolve model '{model_version}' for inference tests: {exc}")
