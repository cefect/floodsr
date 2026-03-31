"""Tests for model CLI commands."""

import json, logging, subprocess, sys
from pathlib import Path

import pytest

from conftest import models_manifest_fp
import floodsr
from floodsr.cli import _parse_arguments, _resolve_log_level


pytestmark = pytest.mark.fast


def _run_cli_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one FloodSR CLI command in a subprocess with explicit stdio capture."""
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "floodsr.cli", *args],
        capture_output=True,
        check=False,
        cwd=repo_root,
        text=True,
    )
    # Replay captured command output so pytest tee-sys and `-s` users can see it.
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed


def test_main_models_list_outputs_model_version(models_manifest_fp: Path):
    """Ensure models list prints version rows."""
    completed = _run_cli_command(["models", "list", "--manifest", str(models_manifest_fp)])
    assert completed.returncode == 0
    assert "v-cli" in completed.stdout


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

    completed = _run_cli_command(args)
    output_fp = Path(completed.stdout.strip())
    assert completed.returncode == 0
    assert output_fp.exists()


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


def test_main_models_fetch_routes_errors_to_stderr(tmp_path: Path):
    """Ensure fetch failures return non-zero exit code and log errors to stderr."""
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

    completed = _run_cli_command(
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
    assert completed.returncode == 1
    assert ("ERROR" in completed.stderr) or ("source model not found" in completed.stderr)


def test_main_version_reports_installed_package_version():
    """Ensure the top-level version flag reports the installed package version."""
    completed = _run_cli_command(["--version"])
    assert completed.returncode == 0
    assert completed.stdout.strip() == f"floodsr {floodsr.__version__}"


def test_main_doctor_reports_runtime_diagnostics():
    """Ensure doctor command reports dependency and provider diagnostics."""
    completed = _run_cli_command(["doctor"])
    assert completed.returncode == 0
    assert f"floodsr_version={floodsr.__version__}" in completed.stdout
    assert "floodsr_module_path=" in completed.stdout
    assert "onnxruntime_installed=" in completed.stdout
    assert "gdal_python_installed=" in completed.stdout
    assert "gdal_vrt_enabled=" in completed.stdout


def test_main_doctor_reports_runtime_diagnostics_json():
    """Ensure doctor command can emit machine-readable JSON diagnostics."""
    completed = _run_cli_command(["doctor", "--json"])
    payload = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert payload["floodsr"]["version"] == floodsr.__version__
    assert payload["floodsr"]["module_path"]
    assert isinstance(payload["gdal"]["vrt_enabled"], bool)
    assert payload["onnxruntime"]["installed"] is True
