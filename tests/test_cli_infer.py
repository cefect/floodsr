"""Tests for infer CLI behavior."""

import os
from pathlib import Path

import numpy as np
import pytest

from floodsr.cli import _parse_arguments, _resolve_default_output_path, _resolve_infer_model_path, main


_BASELINE_INFER_CASES = [
    pytest.param("rss_mersch_A", id="data_case_rss_mersch_a_non_hrdem"),
]

_SPECIAL_INFER_CASES = [
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


@pytest.mark.parametrize("tile_case", _BASELINE_INFER_CASES, indirect=True)
def test_main_infer_runs_data_driven_baseline_case(
    inference_model_fp: Path,
    tmp_path: Path,
    tile_case: dict,
) -> None:
    """Ensure infer command runs for a non-HRDEM data-driven case."""
    pytest.importorskip("onnxruntime")
    rasterio = pytest.importorskip("rasterio")
    case_spec = tile_case["case_spec"]
    tile_dir = tile_case["tile_dir"]
    output_fp = tmp_path / f"{tile_case['case_name']}_pred_cli.tif"

    assert not case_spec["flags"]["in_hrdem"]
    exit_code = main(
        [
            "infer",
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


@pytest.mark.parametrize("tile_case", _SPECIAL_INFER_CASES, indirect=True)
def test_main_infer_runs_in_hrdem_flagged_case(
    inference_model_fp: Path,
    tmp_path: Path,
    tile_case: dict,
) -> None:
    """Ensure infer command runs for in_hrdem-flagged cases."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("rasterio")
    case_spec = tile_case["case_spec"]
    tile_dir = tile_case["tile_dir"]
    output_fp = tmp_path / f"{tile_case['case_name']}_pred_cli_in_hrdem.tif"

    assert case_spec["flags"]["in_hrdem"]
    exit_code = main(
        [
            "infer",
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
    """Ensure infer default output path is generated in cwd with _sr suffix."""
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
    case_spec = tile_case["case_spec"]
    args = _parse_arguments(
        [
            "infer",
            "--in",
            str(tile_case["tile_dir"] / case_spec["inputs"]["lowres_fp"]),
            "--dem",
            str(tile_case["tile_dir"] / case_spec["inputs"]["dem_fp"]),
            "--manifest",
            str(models_manifest_fp),
            "--cache-dir",
            str(cache_dir),
        ]
    )
    model_fp = _resolve_infer_model_path(args)
    assert fetch_exit == 0
    assert isinstance(model_fp, Path)
    assert model_fp.exists()


@pytest.mark.parametrize("tile_case", _FETCH_PARSE_CASES, indirect=True)
def test_parse_infer_allows_fetch_hrdem_without_dem(tile_case: dict):
    """Ensure infer parser accepts --fetch-hrdem without requiring --dem."""
    case_spec = tile_case["case_spec"]
    parsed_args = _parse_arguments(
        [
            "infer",
            "--in",
            str(tile_case["tile_dir"] / case_spec["inputs"]["lowres_fp"]),
            "--fetch-hrdem",
        ]
    )
    assert parsed_args.fetch_hrdem is True
    assert parsed_args.dem is None


@pytest.mark.parametrize("tile_case", _FETCH_PARSE_CASES, indirect=True)
def test_parse_infer_rejects_dem_and_fetch_hrdem_together(tile_case: dict):
    """Ensure infer parser rejects simultaneous --dem and --fetch-hrdem."""
    case_spec = tile_case["case_spec"]
    with pytest.raises(SystemExit):
        _parse_arguments(
            [
                "infer",
                "--in",
                str(tile_case["tile_dir"] / case_spec["inputs"]["lowres_fp"]),
                "--dem",
                str(tile_case["tile_dir"] / case_spec["inputs"]["dem_fp"]),
                "--fetch-hrdem",
            ]
        )


@pytest.mark.parametrize("tile_case", _FETCH_PARSE_CASES, indirect=True)
def test_main_infer_fetch_out_requires_fetch_hrdem(tile_case: dict, tmp_path: Path):
    """Ensure infer runtime rejects --fetch-out unless --fetch-hrdem is enabled."""
    case_spec = tile_case["case_spec"]
    exit_code = main(
        [
            "infer",
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
