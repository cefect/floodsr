"""Tests for model CLI commands."""

import hashlib, json
import logging
from pathlib import Path

import pytest

from floodsr.cli import _parse_arguments, _resolve_log_level, main


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
):
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
    assert "ERROR" in stderr
