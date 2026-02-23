"""Pytest fixtures for FloodSR tests."""

import hashlib, json, logging, pathlib

import numpy as np
import pytest


# Keep case parameterization synced with tests/data/*/case_spec.json.
TEST_TILE_CASES = tuple(
    sorted(case_spec.parent.name for case_spec in pathlib.Path("tests/data").glob("*/case_spec.json"))
)
assert TEST_TILE_CASES, "no data-driven test cases found in tests/data/*/case_spec.json"


def _read_tile_case(case_name: str) -> dict:
    """Load one data-driven test case from tests/data."""
    tile_dir = pathlib.Path("tests/data") / case_name
    case_spec_fp = tile_dir / "case_spec.json"
    assert tile_dir.exists(), f"missing tile directory: {tile_dir}"
    assert case_spec_fp.exists(), f"missing case spec artifact: {case_spec_fp}"
    case_spec = json.loads(case_spec_fp.read_text(encoding="utf-8"))
    assert "model" in case_spec and "inputs" in case_spec and "expected" in case_spec and "flags" in case_spec, (
        f"invalid case spec shape for {case_name}: missing top-level keys"
    )
    assert (
        "lowres_fp" in case_spec["inputs"]
        and "dem_fp" in case_spec["inputs"]
        and "truth_fp" in case_spec["inputs"]
    ), f"invalid case inputs for {case_name}"
    assert "metrics" in case_spec["expected"], f"invalid expected block for {case_name}"
    assert "in_hrdem" in case_spec["flags"], f"missing required flags.in_hrdem for {case_name}"
    return {
        "case_name": case_name,
        "tile_dir": tile_dir,
        "case_spec_fp": case_spec_fp,
        "case_spec": case_spec,
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
@pytest.fixture(scope="session")
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


@pytest.fixture(scope="function")
def models_manifest_fp(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a local one-model manifest fixture for model/CLI tests."""
    source_fp = tmp_path / "source_model.onnx"
    source_fp.write_bytes(b"cli-test-model")
    sha256 = hashlib.sha256(source_fp.read_bytes()).hexdigest()
    manifest = {
        "models": {
            "v-cli": {
                "file_name": "model.onnx",
                "url": source_fp.as_uri(),
                "sha256": sha256,
                "description": "Local CLI test model.",
            }
        }
    }
    manifest_fp = tmp_path / "models.json"
    manifest_fp.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_fp


@pytest.fixture(scope="session")
def tile_case_catalog():
    """Return metadata for all explicitly tracked tile fixtures."""
    return {case_name: _read_tile_case(case_name) for case_name in TEST_TILE_CASES}


@pytest.fixture(scope="session")
def default_model_version():
    """Return the default runnable model version from the packaged manifest."""
    from floodsr.model_registry import list_runnable_model_versions

    runnable_versions = list_runnable_model_versions()
    assert runnable_versions, "manifest has no runnable model versions"
    return runnable_versions[0]


@pytest.fixture
def tile_case(request, tile_case_catalog):
    """Return a tile case by explicit name parameter."""
    case_name = request.param
    assert case_name in tile_case_catalog, f"missing tile case in catalog: {case_name}"
    return tile_case_catalog[case_name]


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
def synthetic_tohr_tiles(tmp_path_factory):
    """Create temporary raster inputs for on-the-fly ToHR coverage tests."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    root = tmp_path_factory.mktemp("tohr_tiles")
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
def tohr_model_fp(tmp_path, default_model_version):
    """Resolve local model path used by engine and CLI ToHR tests."""
    model_version = default_model_version
    local_model_dir = pathlib.Path("_inputs") / model_version
    if local_model_dir.exists():
        local_model_fp_l = sorted(local_model_dir.glob("*.onnx"))
        if local_model_fp_l:
            return local_model_fp_l[0].resolve()

    from floodsr.model_registry import fetch_model

    try:
        return fetch_model(model_version, cache_dir=tmp_path / "cache")
    except Exception as exc:  # pragma: no cover - exercised by test skip behavior
        pytest.skip(f"unable to resolve model '{model_version}' for ToHR tests: {exc}")
