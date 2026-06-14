"""Tests for model CLI commands."""

import json, logging
from pathlib import Path

import pytest

import floodsr
from conftest import models_manifest_fp
from floodsr.cli import _parse_arguments, _resolve_log_level, main


@pytest.mark.fast
def test_main_models_list_outputs_model_version(models_manifest_fp: Path, capsys: pytest.CaptureFixture[str]):
    """Ensure models list prints version rows."""
    exit_code = main(["models", "list", "--manifest", str(models_manifest_fp)])
    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "v-cli" in stdout


@pytest.mark.fast
def test_main_models_list_reports_builtin_model_annotation(capsys: pytest.CaptureFixture[str]):
    """Ensure default manifest listing annotates built-in models."""
    exit_code = main(["models", "list"])
    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "CostGrow_Terrain" in stdout
    assert "(built-in, no download)" in stdout


@pytest.mark.fast
@pytest.mark.parametrize(
    "cli_args, expected_level",
    [
        pytest.param([], logging.INFO, id="default_info_level"),
        pytest.param(["-v", "-v"], logging.DEBUG, id="verbose_twice_to_debug"),
        pytest.param(["-q", "-q"], logging.ERROR, id="quiet_twice_to_error"),
    ],
)
def test_resolve_log_level_from_cli_arguments(cli_args: list[str], expected_level: int):
    """Ensure CLI logging defaults and verbosity flags resolve effective levels."""
    parsed_args = _parse_arguments([*cli_args, "models", "list"])
    resolved_level = _resolve_log_level(parsed_args)
    assert isinstance(resolved_level, int)
    assert resolved_level == expected_level


@pytest.mark.fast
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
    """Ensure models fetch prints one-line cache metadata for fetched weights."""
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
    stdout = capsys.readouterr().out.strip()
    assert exit_code == 0
    assert "version=v-cli " in stdout
    assert " stored=" in stdout
    assert " retrieved_from=" in stdout
    output_fp = Path(stdout.split(" stored=", 1)[1].split(" retrieved_from=", 1)[0])
    assert output_fp.exists()


@pytest.mark.fast
def test_main_models_fetch_prints_cache_hit_summary(
    tmp_path: Path,
    models_manifest_fp: Path,
    capsys: pytest.CaptureFixture[str],
):
    """Ensure models fetch still reports cache-vs-fetch source on repeat calls."""
    args = [
        "models",
        "fetch",
        "v-cli",
        "--manifest",
        str(models_manifest_fp),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--no-progress",
    ]
    first_exit = main(args)
    first_stdout = capsys.readouterr().out.strip()
    second_exit = main(args)
    second_stdout = capsys.readouterr().out.strip()
    assert first_exit == 0
    assert second_exit == 0
    assert " retrieved_from=fetch:file" in first_stdout
    assert " retrieved_from=cache" in second_stdout


@pytest.mark.fast
@pytest.mark.parametrize(
    "progress_flag, expected_show_progress",
    [
        pytest.param("--show-progress", True, id="explicit_show_progress"),
        pytest.param("--no-progress", False, id="explicit_no_progress"),
    ],
)
def test_parse_models_fetch_progress_flags(progress_flag: str, expected_show_progress: bool):
    """Ensure models fetch exposes the documented progress flag pair."""
    parsed_args = _parse_arguments(["models", "fetch", "v-cli", progress_flag])
    assert isinstance(parsed_args.show_progress, bool)
    assert parsed_args.show_progress is expected_show_progress


@pytest.mark.fast
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


@pytest.mark.fast
def test_main_models_fetch_rejects_builtin_model(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    """Ensure CLI fetch rejects built-in models cleanly."""
    caplog.set_level(logging.ERROR)
    exit_code = main(["models", "fetch", "CostGrow_Terrain"])
    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert ("built-in" in stderr) or ("built-in" in caplog.text)


@pytest.mark.fast
def test_main_version_reports_installed_package_version(capsys: pytest.CaptureFixture[str]):
    """Ensure the top-level version flag reports the installed package version."""
    exit_code = main(["--version"])
    stdout = capsys.readouterr().out.strip()
    assert exit_code == 0
    assert stdout == f"floodsr {floodsr.__version__}"


@pytest.mark.fast
def test_main_doctor_reports_runtime_diagnostics(capsys: pytest.CaptureFixture[str]):
    """Ensure doctor command reports dependency and provider diagnostics."""
    exit_code = main(["doctor"])
    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert f"floodsr_version={floodsr.__version__}" in stdout
    assert "floodsr_module_path=" in stdout
    assert "onnxruntime_installed=" in stdout
    assert "gdal_python_installed=" in stdout
    assert "gdal_vrt_enabled=" in stdout
    assert "pcraster_installed=" in stdout
    assert "pcraster_spreadzone_available=" in stdout


@pytest.mark.fast
def test_main_doctor_reports_runtime_diagnostics_json(capsys: pytest.CaptureFixture[str]):
    """Ensure doctor command can emit machine-readable JSON diagnostics."""
    exit_code = main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["floodsr"]["version"] == floodsr.__version__
    assert payload["floodsr"]["module_path"]
    assert isinstance(payload["gdal"]["vrt_enabled"], bool)
    assert payload["onnxruntime"]["installed"] is True
    assert isinstance(payload["pcraster"]["installed"], bool)
    assert isinstance(payload["pcraster"]["spreadzone_available"], bool)
