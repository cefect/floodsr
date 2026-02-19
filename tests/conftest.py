
import os, logging, sys, hashlib, json
import pytest, yaml
import pathlib
import pandas as pd
 
 
 

# project parametesr
# Get the project root (parent of tests directory)
project_root = pathlib.Path(__file__).parent.parent
 

 
#===============================================================================
# pytest custom config------------
#===============================================================================
 

def pytest_runtest_teardown(item, nextitem):
    """custom teardown message"""
    test_name = item.name
    print(f"\n{'='*20} Test completed: {test_name} {'='*20}\n\n\n")
    
def pytest_report_header(config):
    """modifies the pytest header to show all of the arguments"""
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
def tile_data_dir():
    """Return the shared tile test-data directory."""
    tile_dir = pathlib.Path("tests/data/2407_FHIMP_tile")
    assert tile_dir.exists(), f"missing tile data directory: {tile_dir}"
    return tile_dir


@pytest.fixture(scope="session")
def tile_metrics_fp(tile_data_dir):
    """Return the notebook-derived metric artifact path."""
    metrics_fp = tile_data_dir / "metrics.json"
    assert metrics_fp.exists(), f"missing metric artifact file: {metrics_fp}"
    return metrics_fp


_TILE_METRICS_FILES = sorted(pathlib.Path("tests/data").glob("*/metrics.json"))


@pytest.fixture(
    scope="function",
    params=_TILE_METRICS_FILES or [None],
    ids=lambda v: v.parent.name if v is not None else "no_tile_metrics",
)
def tile_case(request):
    """Return one tile case discovered from tests/data/*/metrics.json."""
    metrics_fp = request.param
    if metrics_fp is None:
        pytest.skip("no tile metrics artifacts found under tests/data/*/metrics.json")
    tile_dir = metrics_fp.parent
    artifact = json.loads(metrics_fp.read_text(encoding="utf-8"))
    return {"tile_dir": tile_dir, "metrics_fp": metrics_fp, "artifact": artifact}


@pytest.fixture(scope="function")
def phase23_model_fp(tmp_path):
    """Resolve local model path used by Phase 2/3 engine and CLI tests."""
    model_version = "4690176_0_1770580046_train_base_16"
    local_fp = pathlib.Path("_inputs") / model_version / "model_infer.onnx"
    if local_fp.exists():
        return local_fp.resolve()

    from floodsr.model_registry import fetch_model

    try:
        return fetch_model(model_version, cache_dir=tmp_path / "cache")
    except Exception as exc:
        pytest.skip(f"unable to resolve model '{model_version}' for inference tests: {exc}")

 
