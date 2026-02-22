"""Tests for ToHR CLI behavior."""

import hashlib, json, os
from pathlib import Path

import numpy as np
import pytest

from floodsr.cli import _parse_arguments, _resolve_default_output_path, _resolve_tohr_model_spec, main


_BASELINE_TOHR_CASES = [
    pytest.param("rss_mersch_A", id="data_case_rss_mersch_a_non_hrdem"),
]

_SPECIAL_TOHR_CASES = [
    pytest.param("2407_FHIMP_tile", id="data_case_2407_fhimp_tile_in_hrdem"),
]

_DEFAULT_OUTPUT_CASES = [
    pytest.param("2407_FHIMP_tile", id="data_case_output_name_2407_fhimp_tile"),
    pytest.param("rss_mersch_A", id="data_case_output_name_rss_mersch_a"),
]

_RESOLVE_MODEL_CASES = [
    pytest.param("2407_FHIMP_tile", id="data_case_resolve_model_2407_fhimp_tile"),
]

_FETCH_PARSE_CASES = [
    pytest.param("2407_FHIMP_tile", id="data_case_fetch_parse_2407_fhimp_tile"),
]


@pytest.mark.parametrize("tile_case", _BASELINE_TOHR_CASES, indirect=True)
def test_main_tohr_runs_data_driven_baseline_case(
    inference_model_fp: Path,
    tmp_path: Path,
    tile_case: dict,
) -> None:
    """Ensure tohr command runs for a non-HRDEM data-driven case."""
    pytest.importorskip("onnxruntime")
    rasterio = pytest.importorskip("rasterio")
    case_spec = tile_case["case_spec"]
    tile_dir = tile_case["tile_dir"]
    output_fp = tmp_path / f"{tile_case['case_name']}_pred_cli.tif"

    assert not case_spec["flags"]["in_hrdem"]
    exit_code = main(
        [
            "tohr",
            "--in",
            str(tile_dir / case_spec["inputs"]["lowres_fp"]),
            "--dem",
            str(tile_dir / case_spec["inputs"]["dem_fp"]),
            "--out",
            str(output_fp),
            "--model-path",
            str(inference_model_fp),
        ]
    )
    with rasterio.open(output_fp) as ds:
        pred = ds.read(1)

    assert exit_code == 0
    assert pred.dtype == np.float32
    assert pred.size > 0


@pytest.mark.parametrize("tile_case", _SPECIAL_TOHR_CASES, indirect=True)
def test_main_tohr_runs_in_hrdem_flagged_case(
    inference_model_fp: Path,
    tmp_path: Path,
    tile_case: dict,
) -> None:
    """Ensure tohr command runs for in_hrdem-flagged cases."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("rasterio")
    case_spec = tile_case["case_spec"]
    tile_dir = tile_case["tile_dir"]
    output_fp = tmp_path / f"{tile_case['case_name']}_pred_cli_in_hrdem.tif"

    assert case_spec["flags"]["in_hrdem"]
    exit_code = main(
        [
            "tohr",
            "--in",
            str(tile_dir / case_spec["inputs"]["lowres_fp"]),
            "--dem",
            str(tile_dir / case_spec["inputs"]["dem_fp"]),
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


@pytest.mark.parametrize("tile_case", _DEFAULT_OUTPUT_CASES, indirect=True)
def test_default_output_path_uses_cwd_and_input_stem(tmp_path: Path, tile_case: dict):
    """Ensure ToHR default output path is generated in cwd with _sr suffix."""
    case_spec = tile_case["case_spec"]
    tile_dir = tile_case["tile_dir"]
    input_fp = tile_dir / case_spec["inputs"]["lowres_fp"]
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        output_fp = _resolve_default_output_path(input_fp)
    finally:
        os.chdir(cwd)
    assert isinstance(output_fp, Path)
    assert output_fp == (tmp_path / f"{input_fp.stem}_sr.tif").resolve()


@pytest.mark.parametrize("tile_case", _RESOLVE_MODEL_CASES, indirect=True)
def test_resolve_tohr_model_spec_uses_cached_manifest_default(tmp_path: Path, tile_case: dict):
    """Ensure ToHR default model resolution uses cached first runnable manifest model."""
    model_version = "4690176_0_1770580046_train_base_16"
    source_fp = tmp_path / "source_model.onnx"
    source_fp.write_bytes(b"cli-test-model")
    source_sha256 = hashlib.sha256(source_fp.read_bytes()).hexdigest()
    manifest_payload = {
        "models": {
            model_version: {
                "file_name": "model_infer.onnx",
                "url": source_fp.as_uri(),
                "sha256": source_sha256,
                "description": "Runnable local model for ToHR CLI model resolution tests.",
            }
        }
    }
    manifest_fp = tmp_path / "models_tohr.json"
    manifest_fp.write_text(json.dumps(manifest_payload), encoding="utf-8")

    cache_dir = tmp_path / "cache"
    fetch_exit = main(
        [
            "models",
            "fetch",
            model_version,
            "--manifest",
            str(manifest_fp),
            "--cache-dir",
            str(cache_dir),
        ]
    )
    case_spec = tile_case["case_spec"]
    args = _parse_arguments(
        [
            "tohr",
            "--in",
            str(tile_case["tile_dir"] / case_spec["inputs"]["lowres_fp"]),
            "--dem",
            str(tile_case["tile_dir"] / case_spec["inputs"]["dem_fp"]),
            "--manifest",
            str(manifest_fp),
            "--cache-dir",
            str(cache_dir),
        ]
    )
    resolved_version, model_fp = _resolve_tohr_model_spec(args)
    assert fetch_exit == 0
    assert resolved_version == model_version
    assert model_fp.exists()


@pytest.mark.parametrize("tile_case", _FETCH_PARSE_CASES, indirect=True)
def test_parse_tohr_allows_fetch_hrdem_without_dem(tile_case: dict):
    """Ensure tohr parser accepts --fetch-hrdem without requiring --dem."""
    case_spec = tile_case["case_spec"]
    parsed_args = _parse_arguments(
        [
            "tohr",
            "--in",
            str(tile_case["tile_dir"] / case_spec["inputs"]["lowres_fp"]),
            "--fetch-hrdem",
        ]
    )
    assert parsed_args.fetch_hrdem is True
    assert parsed_args.dem is None


@pytest.mark.parametrize("tile_case", _FETCH_PARSE_CASES, indirect=True)
def test_parse_tohr_rejects_dem_and_fetch_hrdem_together(tile_case: dict):
    """Ensure tohr parser rejects simultaneous --dem and --fetch-hrdem."""
    case_spec = tile_case["case_spec"]
    with pytest.raises(SystemExit):
        _parse_arguments(
            [
                "tohr",
                "--in",
                str(tile_case["tile_dir"] / case_spec["inputs"]["lowres_fp"]),
                "--dem",
                str(tile_case["tile_dir"] / case_spec["inputs"]["dem_fp"]),
                "--fetch-hrdem",
            ]
        )


@pytest.mark.parametrize("tile_case", _FETCH_PARSE_CASES, indirect=True)
def test_main_tohr_fetch_out_requires_fetch_hrdem(tile_case: dict, tmp_path: Path):
    """Ensure tohr runtime rejects --fetch-out unless --fetch-hrdem is enabled."""
    case_spec = tile_case["case_spec"]
    exit_code = main(
        [
            "tohr",
            "--in",
            str(tile_case["tile_dir"] / case_spec["inputs"]["lowres_fp"]),
            "--dem",
            str(tile_case["tile_dir"] / case_spec["inputs"]["dem_fp"]),
            "--fetch-out",
            str(tmp_path / "fetched_dem.tif"),
        ]
    )
    assert isinstance(exit_code, int)
    assert exit_code == 1
