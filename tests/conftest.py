"""Pytest fixtures for FloodSR tests."""

import hashlib, json, logging, pathlib, sys

import pytest


# Keep case parameterization synced with tests/data/*/case_spec.json.
TEST_TILE_CASES = tuple(
    sorted(case_spec.parent.name for case_spec in pathlib.Path("tests/data").glob("*/case_spec.json"))
)
assert TEST_TILE_CASES, "no data-driven test cases found in tests/data/*/case_spec.json"
LOCAL_TILE_CASES = tuple(case_name for case_name in TEST_TILE_CASES if case_name.startswith(("fathom", "rss_")))


def _get_numpy():
    """Import numpy only for fixtures that actually need array helpers."""
    # Keep conftest import-light so docs-only environments do not need runtime deps.
    # Do not add unconditional third-party imports here unless every test env ships them.
    return pytest.importorskip("numpy", reason="numpy not detected in environment.")


def _read_tile_case(case_name: str) -> dict:
    """Load one data-driven test case from tests/data."""
    tile_dir = pathlib.Path("tests/data") / case_name
    case_spec_fp = tile_dir / "case_spec.json"
    assert tile_dir.exists(), f"missing tile directory: {tile_dir}"
    assert case_spec_fp.exists(), f"missing case spec artifact: {case_spec_fp}"
    case_spec = json.loads(case_spec_fp.read_text(encoding="utf-8"))
    assert "inputs" in case_spec and "flags" in case_spec, (
        f"invalid case spec shape for {case_name}: missing top-level keys"
    )
    assert (
        "lowres_fp" in case_spec["inputs"]
        and "dem_fp" in case_spec["inputs"]
        and "truth_fp" in case_spec["inputs"]
    ), f"invalid case inputs for {case_name}"
    # Validate configured input paths, but allow explicit `False` sentinels for optional inputs.
    for input_key in ("lowres_fp", "dem_fp", "truth_fp"):
        input_value = case_spec["inputs"][input_key]
        if input_value is False:
            continue
        assert isinstance(input_value, str) and input_value.strip(), (
            f"invalid case input value for {case_name}/{input_key}: {input_value!r}"
        )
        assert (tile_dir / input_value).exists(), (
            f"missing case input file for {case_name}/{input_key}:\n    {tile_dir / input_value}"
        )
    assert "in_hrdem" in case_spec["flags"], f"missing required flags.in_hrdem for {case_name}"
    if "supports_regression_metrics" in case_spec["flags"]:
        assert isinstance(case_spec["flags"]["supports_regression_metrics"], bool), (
            f"invalid flags.supports_regression_metrics for {case_name}"
        )
    requires_regression_metrics = (
        case_spec["inputs"]["truth_fp"] is not False
        and bool(case_spec["flags"].get("supports_regression_metrics", True))
    )
    if requires_regression_metrics:
        assert "expected" in case_spec, f"missing expected block for regression case {case_name}"
        assert isinstance(case_spec["expected"], dict) and case_spec["expected"], f"invalid expected block for {case_name}"
        for run_label, run_spec in case_spec["expected"].items():
            assert "params" in run_spec and "metrics" in run_spec, f"invalid expected run block for {case_name}/{run_label}"
            assert isinstance(run_spec["params"], dict), f"invalid params block for {case_name}/{run_label}"
            assert "model_version" in run_spec["params"], f"missing params.model_version for {case_name}/{run_label}"
            assert isinstance(run_spec["metrics"], dict), f"invalid metrics block for {case_name}/{run_label}"
            assert (
                "mase_m" in run_spec["metrics"] and "rmse_m" in run_spec["metrics"] and "ssim" in run_spec["metrics"]
            ), f"missing expected metrics keys for {case_name}/{run_label}"
    return {
        "case_name": case_name,
        "tile_dir": tile_dir,
        "case_spec_fp": case_spec_fp,
        "case_spec": case_spec,
    }


def _write_single_band_geotiff(fp: pathlib.Path, array, transform, crs: str, nodata: float = -9999.0) -> None:
    """Write a one-band float32 GeoTIFF with deterministic defaults."""
    import rasterio

    np = _get_numpy()
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
# pytest custom config
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
def logger(tmp_path_factory):
    """Simple logger fixture for the function under test."""
    log = logging.getLogger("pytest")
    log.setLevel(logging.DEBUG)
    # Write pytest logger output to a stable per-session file in pytest temp output.
    log_dir = tmp_path_factory.mktemp("test_logs")
    log_fp = log_dir / "pytest.session.log"
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    # keep handlers minimal to avoid duplicate logs across runs
    if not any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) for handler in log.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        log.addHandler(stream_handler)
    if not any(isinstance(handler, logging.FileHandler) and pathlib.Path(handler.baseFilename) == log_fp for handler in log.handlers):
        file_handler = logging.FileHandler(log_fp)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
    log.info(f"pytest logger file:\n    {log_fp}")
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
    """Return metadata for all explicitly tracked tile fixtures.

    Skips cases with missing input files (e.g., proprietary test data).
    These are marked @pytest.mark.local and only run when all data is present.
    """
    catalog = {}
    for case_name in TEST_TILE_CASES:
        try:
            catalog[case_name] = _read_tile_case(case_name)
        except AssertionError as e:
            if "missing case input file" in str(e):
                # Skip cases with missing proprietary test data
                logging.getLogger("pytest").debug(f"Skipping case {case_name}: missing data file(s)")
                continue
            raise
    return catalog


@pytest.fixture(scope="session")
def default_model_version():
    """Return the default runnable model version from the packaged manifest."""
    from floodsr.model_registry import list_runnable_model_versions

    runnable_versions = list_runnable_model_versions()
    assert runnable_versions, "manifest has no runnable model versions"
    return runnable_versions[0]


@pytest.fixture
def tile_case_d(case_id, tile_case_catalog):
    """Return one tile case payload by explicit case_id parameter.

    Skips test if the case has missing proprietary test data files.
    """
    assert isinstance(case_id, str) and case_id.strip(), f"invalid case_id parameter: {case_id!r}"
    if case_id not in tile_case_catalog:
        pytest.skip(f"test case '{case_id}' requires proprietary data not in repository")
    return tile_case_catalog[case_id]


@pytest.fixture(scope="session")
def ort_tile_inputs():
    """Create synthetic arrays that match a single model tile size."""
    np = _get_numpy()
    return {
        "depth_lr": np.full((32, 32), 1.5, dtype=np.float32),
        "dem_hr": np.linspace(500.0, 1000.0, 512 * 512, dtype=np.float32).reshape((512, 512)),
        "depth_lr_nodata": -9999.0,
        "dem_hr_nodata": -9999.0,
    }


@pytest.fixture(scope="session")
def synthetic_tohr_tiles(tmp_path_factory):
    """Create temporary raster inputs for on-the-fly ToHR coverage tests."""
    np = _get_numpy()
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


@pytest.fixture(scope="session")
def synthetic_tohr_windowed_tiles(tmp_path_factory):
    """Create a large-output synthetic case that should trigger windowed hard IO."""
    np = _get_numpy()
    pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    root = tmp_path_factory.mktemp("tohr_windowed_tiles")
    lr_shape = (32, 32)
    hr_shape = (4096, 4096)
    crs = "EPSG:32633"
    hr_resolution = 2.0
    lr_resolution = (hr_shape[0] * hr_resolution) / lr_shape[0]
    x0, y0 = 500000.0, 4100000.0

    depth_lr = np.full(lr_shape, 1.0, dtype=np.float32)
    dem = np.tile(np.linspace(500.0, 1000.0, hr_shape[1], dtype=np.float32), (hr_shape[0], 1))

    depth_lr_fp = root / "depth_lr_windowed.tif"
    dem_fp = root / "dem_windowed.tif"
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
        "output_fp": root / "pred_sr_windowed.tif",
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
