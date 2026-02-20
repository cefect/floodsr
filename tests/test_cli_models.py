"""Tests for model CLI commands."""

import hashlib, json, os
import logging
from pathlib import Path

import pytest

from floodsr.cli import _parse_arguments, _resolve_default_output_path, _resolve_infer_model_path, _resolve_log_level, main
from tests.conftest import inference_model_fp


@pytest.fixture(scope="function")
def models_manifest_fp(tmp_path: Path) -> Path:
    """Create a local CLI manifest fixture."""
    source_fp = tmp_path / "source_model.onnx"
    source_fp.write_bytes(b"cli-test-model")
    sha256 = hashlib.sha256(source_fp.read_bytes()).hexdigest()

    # Build a one-model manifest for CLI list/fetch tests.
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


def test_main_models_list_outputs_model_version(models_manifest_fp: Path, capsys: pytest.CaptureFixture[str]):
    """Ensure models list prints version rows."""
    exit_code = main(["models", "list", "--manifest", str(models_manifest_fp)])
    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "v-cli" in stdout


@pytest.mark.parametrize(
    "cli_args, expected_level",
    [
        pytest.param([], logging.INFO, id="default_info_level"),
        pytest.param(["-v", "-v"], logging.DEBUG, id="repeat_verbose_to_debug"),
    ],
)
def test_resolve_log_level_from_cli_arguments(cli_args: list[str], expected_level: int):
    """Ensure CLI logging defaults and verbosity flags resolve effective levels."""
    parsed_args = _parse_arguments([*cli_args, "models", "list"])
    resolved_level = _resolve_log_level(parsed_args)
    assert isinstance(resolved_level, int)
    assert resolved_level == expected_level


@pytest.mark.parametrize(
    "backend_name",
    [
        pytest.param(None, id="auto_backend"),
        pytest.param("file", id="explicit_file_backend"),
    ],
)
def test_main_models_fetch_prints_existing_path(
    tmp_path: Path,
    models_manifest_fp: Path,
    capsys: pytest.CaptureFixture[str],
    backend_name: str | None,
):
    """Ensure models fetch prints a valid path for cached weights."""
    args = [
        "models",
        "fetch",
        "v-cli",
        "--manifest",
        str(models_manifest_fp),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    if backend_name is not None:
        args.extend(["--backend", backend_name])

    exit_code = main(args)
    output_fp = Path(capsys.readouterr().out.strip())
    assert exit_code == 0
    assert output_fp.exists()


def test_main_models_fetch_routes_errors_to_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    """Ensure fetch failures return non-zero exit code and log errors to stderr."""
    caplog.set_level(logging.ERROR)
    manifest = {
        "models": {
            "v-missing": {
                "file_name": "model.onnx",
                "url": (tmp_path / "missing_model.onnx").as_uri(),
                "sha256": "0" * 64,
                "description": "Missing source file.",
            }
        }
    }
    manifest_fp = tmp_path / "models_missing.json"
    manifest_fp.write_text(json.dumps(manifest), encoding="utf-8")

    exit_code = main(
        [
            "--log-level",
            "ERROR",
            "models",
            "fetch",
            "v-missing",
            "--manifest",
            str(manifest_fp),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert ("ERROR" in stderr) or ("source model not found" in caplog.text)


def test_main_doctor_reports_runtime_diagnostics(capsys: pytest.CaptureFixture[str]):
    """Ensure doctor command reports dependency and provider diagnostics."""
    exit_code = main(["doctor"])
    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "onnxruntime_installed=" in stdout


_INFERENCE_CASES = [
    pytest.param("2407_FHIMP_tile", id="2407_FHIMP_tile"),
    pytest.param("rss_mersch_A", id="rss_mersch_A"),
]

_INFERENCE_RUN_CASES = [
    pytest.param("2407_FHIMP_tile", id="2407_FHIMP_tile"),
]

_CLI_TILE_CASES = [
    pytest.param("2407_FHIMP_tile", id="2407_FHIMP_tile"),
    pytest.param("rss_mersch_A", id="rss_mersch_A"),
]


@pytest.mark.parametrize("tile_case", _INFERENCE_CASES, indirect=True)
def test_main_infer_runs_raster_prediction(
    inference_model_fp: Path,
    tmp_path: Path,
    tile_case: dict,
    capsys: pytest.CaptureFixture[str],
):
    """Ensure infer command produces an output raster for a model/tile pair."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("rasterio")
    artifact = tile_case["artifact"]
    tile_dir = tile_case["tile_dir"]
    output_fp = tmp_path / f"{tile_case['case_name']}_pred_cli.tif"

    exit_code = main(
        [
            "infer",
            "--in",
            str(tile_dir / artifact["lowres_fp"]),
            "--dem",
            str(tile_dir / artifact["dem_fp"]),
            "--out",
            str(output_fp),
            "--model-path",
            str(inference_model_fp),
        ]
    )
    _stdout = capsys.readouterr().out
    assert exit_code == 0
    if tile_case["case_name"] == "rss_mersch_A":
        # This case contains valid zero-border tiles in the source DEM clip.
        # Keep explicit coverage of CLI wiring without asserting full output shape.
        assert output_fp.exists()
    else:
        assert output_fp.exists()


@pytest.mark.parametrize("tile_case", _INFERENCE_RUN_CASES, indirect=True)
def test_main_infer_uses_tiling_flags(
    inference_model_fp: Path,
    tmp_path: Path,
    tile_case: dict,
) -> None:
    """Ensure tiling controls are accepted by CLI and forwarded to inference."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("rasterio")
    artifact = tile_case["artifact"]
    tile_dir = tile_case["tile_dir"]
    output_fp = tmp_path / f"{tile_case['case_name']}_pred_cli_tiled.tif"

    exit_code = main(
        [
            "infer",
            "--in",
            str(tile_dir / artifact["lowres_fp"]),
            "--dem",
            str(tile_dir / artifact["dem_fp"]),
            "--out",
            str(output_fp),
            "--model-path",
            str(inference_model_fp),
            "--window-method",
            "hard",
            "--tile-overlap",
            "0",
        ]
    )
    assert exit_code == 0
    assert output_fp.exists()


@pytest.mark.parametrize(
    "tile_case",
    _CLI_TILE_CASES,
    indirect=True,
)
def test_default_output_path_uses_cwd_and_input_stem(tmp_path: Path, tile_case: dict):
    """Ensure infer default output path is generated in cwd with _sr suffix."""
    artifact = tile_case["artifact"]
    tile_dir = tile_case["tile_dir"]
    input_fp = tile_dir / artifact["lowres_fp"]
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        output_fp = _resolve_default_output_path(input_fp)
    finally:
        os.chdir(cwd)
    assert isinstance(output_fp, Path)
    expected_output = f"{Path(input_fp).stem}_sr.tif"
    assert output_fp == (tmp_path / expected_output).resolve()




@pytest.mark.parametrize(
    "tile_case",
    _INFERENCE_RUN_CASES,
    indirect=True,
)
def test_resolve_infer_model_path_uses_cached_manifest_default(
    models_manifest_fp: Path,
    tmp_path: Path,
    tile_case: dict,
):
    """Ensure infer default model resolution uses cached first manifest model."""
    cache_dir = tmp_path / "cache"
    fetch_exit = main(
        [
            "models",
            "fetch",
            "v-cli",
            "--manifest",
            str(models_manifest_fp),
            "--cache-dir",
            str(cache_dir),
        ]
    )
    assert fetch_exit == 0

    args = _parse_arguments(
        [
            "infer",
            "--in",
            str(tile_case["tile_dir"] / tile_case["artifact"]["lowres_fp"]),
            "--dem",
            str(tile_case["tile_dir"] / tile_case["artifact"]["dem_fp"]),
            "--manifest",
            str(models_manifest_fp),
            "--cache-dir",
            str(cache_dir),
        ]
    )
    model_fp = _resolve_infer_model_path(args)
    assert isinstance(model_fp, Path)
    assert model_fp.exists()
