"""Tests for ToHR CLI behavior."""

import hashlib, json, os
from pathlib import Path

import pytest
np = pytest.importorskip("numpy")

import floodsr.models.CostGrow_Terrain as costgrow_module
import floodsr.tohr
from conftest import default_model_version, tile_case_d, tohr_model_fp
from floodsr.cli import _parse_arguments, _resolve_default_output_path, _resolve_tohr_model_spec, main

pytestmark = pytest.mark.local

@pytest.mark.parametrize(
    "case_id",
    [
        pytest.param("rss_dudelange_A", id="data_case_rss_dudelange_a_non_hrdem", marks=pytest.mark.local),
        pytest.param("rss_mersch_A", id="data_case_rss_mersch_a_non_hrdem", marks=pytest.mark.local),
    ],
)
@pytest.mark.e2e
@pytest.mark.network
def test_main_tohr_runs_data_driven_baseline_case(
    tohr_model_fp: Path,
    tmp_path: Path,
    tile_case_d: dict,
) -> None:
    """Ensure tohr command runs for a non-HRDEM data-driven case."""
    pytest.importorskip("onnxruntime")
    rasterio = pytest.importorskip("rasterio")
    case_spec = tile_case_d["case_spec"]
    tile_dir = tile_case_d["tile_dir"]
    output_fp = tmp_path / f"{tile_case_d['case_name']}_pred_cli.tif"

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
            str(tohr_model_fp),
        ]
    )
    with rasterio.open(output_fp) as ds:
        pred = ds.read(1)

    assert exit_code == 0
    assert pred.dtype == np.float32
    assert pred.size > 0


@pytest.mark.parametrize(
    "case_id",
    [
        pytest.param("2407_FHIMP_tile", id="data_case_2407_fhimp_tile_in_hrdem"),
        pytest.param("fathom_clip", id="data_case_fathom_clip_in_hrdem", marks=pytest.mark.local),
        #pytest.param("fathom_n51w115", id="data_case_fathom_n51w115_in_hrdem", marks=pytest.mark.local),
    ],
)
@pytest.mark.e2e
@pytest.mark.network
def test_main_tohr_runs_in_hrdem_flagged_case(
    tohr_model_fp: Path,
    tmp_path: Path,
    tile_case_d: dict,
):
    """Ensure tohr command runs for in_hrdem-flagged cases."""
    pytest.importorskip("onnxruntime")
    rasterio = pytest.importorskip("rasterio")
    case_spec = tile_case_d["case_spec"]
    tile_dir = tile_case_d["tile_dir"]
    output_fp = tmp_path / f"{tile_case_d['case_name']}_pred_cli_in_hrdem.tif"

    assert case_spec["flags"]["in_hrdem"]
    cli_args = [
        "tohr",
        "--in",
        str(tile_dir / case_spec["inputs"]["lowres_fp"]),
        "--out",
        str(output_fp),
        "--model-path",
        str(tohr_model_fp),
        "--window-method",
        "hard",
        "--tile-overlap",
        "0",
    ]
    if case_spec["inputs"]["dem_fp"] is False:
        cli_args.extend(["--fetch-hrdem", "--crs-policy", "use-dem"])
    else:
        cli_args.extend(["--dem", str(tile_dir / case_spec["inputs"]["dem_fp"])])
    exit_code = main(cli_args)
    with rasterio.open(output_fp) as ds:
        pred = ds.read(1)

    assert exit_code == 0
    assert pred.dtype == np.float32
    assert pred.size > 0


@pytest.mark.parametrize("case_id", [pytest.param("fathom_clip", id="tutorial_3_like_fetch_force_tiling_case", marks=pytest.mark.local)])
@pytest.mark.e2e
@pytest.mark.network
def test_main_tohr_runs_tutorial_3_like_fetch_force_tiling_case(
    tohr_model_fp: Path,
    tmp_path: Path,
    tile_case_d: dict,
):
    """Ensure the Tutorial 3-style CLI command shape runs on the small fetched-HRDEM fixture."""
    pytest.importorskip("onnxruntime")
    rasterio = pytest.importorskip("rasterio")
    case_spec = tile_case_d["case_spec"]
    tile_dir = tile_case_d["tile_dir"]
    output_fp = tmp_path / f"{tile_case_d['case_name']}_tutorial_3_like_pred.tif"
    fetched_dem_fp = tmp_path / f"{tile_case_d['case_name']}_tutorial_3_like_fetch_dem.vrt"
    cache_dir = tmp_path / "tutorial_3_like_cache"

    assert case_spec["flags"]["in_hrdem"]
    assert case_spec["inputs"]["dem_fp"] is False
    exit_code = main(
        [
            "tohr",
            "--in",
            str(tile_dir / case_spec["inputs"]["lowres_fp"]),
            "--fetch-hrdem",
            "--fetch-out",
            str(fetched_dem_fp),
            "--fetch-force-tiling",
            "--cache-dir",
            str(cache_dir),
            "--crs-policy",
            "use-dem",
            "--model-path",
            str(tohr_model_fp),
            "--window-method",
            "hard",
            "--tile-overlap",
            "0",
            "--out",
            str(output_fp),
            "--min-depth-threshold",
            "0.1",
        ]
    )
    with rasterio.open(output_fp) as ds:
        pred = ds.read(1)

    assert exit_code == 0
    assert fetched_dem_fp.exists()
    assert pred.dtype == np.float32
    assert pred.size > 0


@pytest.mark.parametrize(
    "case_id",
    [
        pytest.param("2407_FHIMP_tile", id="data_case_output_name_2407_fhimp_tile"),
        pytest.param("fathom_clip", id="data_case_output_name_fathom_clip", marks=pytest.mark.local),
        #pytest.param("fathom_n51w115", id="data_case_output_name_fathom_n51w115", marks=pytest.mark.local),
        pytest.param("rss_dudelange_A", id="data_case_output_name_rss_dudelange_a", marks=pytest.mark.local),
        pytest.param("rss_mersch_A", id="data_case_output_name_rss_mersch_a", marks=pytest.mark.local),
    ],
)
@pytest.mark.fast
def test_default_output_path_uses_cwd_and_input_stem(tmp_path: Path, tile_case_d: dict):
    """Ensure ToHR default output path is generated in cwd with _sr suffix."""
    case_spec = tile_case_d["case_spec"]
    tile_dir = tile_case_d["tile_dir"]
    input_fp = tile_dir / case_spec["inputs"]["lowres_fp"]
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        output_fp = _resolve_default_output_path(input_fp)
    finally:
        os.chdir(cwd)
    assert isinstance(output_fp, Path)
    assert output_fp == (tmp_path / f"{input_fp.stem}_sr.tif").resolve()


@pytest.mark.parametrize("case_id", [pytest.param("2407_FHIMP_tile", id="data_case_resolve_model_2407_fhimp_tile")])
@pytest.mark.fast
def test_resolve_tohr_model_spec_uses_cached_manifest_default(
    tmp_path: Path,
    tile_case_d: dict,
    default_model_version: str,
):
    """Ensure ToHR default model resolution uses cached first runnable manifest model."""
    model_version = default_model_version
    source_fp = tmp_path / "source_model.onnx"
    source_fp.write_bytes(b"cli-test-model")
    source_sha256 = hashlib.sha256(source_fp.read_bytes()).hexdigest()
    manifest_payload = {
        "models": {
            model_version: {
                "file_name": "model_tohr.onnx",
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
    case_spec = tile_case_d["case_spec"]
    args = _parse_arguments(
        [
            "tohr",
            "--in",
            str(tile_case_d["tile_dir"] / case_spec["inputs"]["lowres_fp"]),
            "--dem",
            str(tile_case_d["tile_dir"] / case_spec["inputs"]["dem_fp"]),
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


@pytest.mark.fast
def test_resolve_tohr_model_spec_returns_none_for_builtin_model():
    """Ensure built-in models resolve without downloading artifacts."""
    args = _parse_arguments(
        [
            "tohr",
            "--in",
            "tests/data/2407_FHIMP_tile/lowres032.tif",
            "--dem",
            "tests/data/2407_FHIMP_tile/hires002_dem.tif",
            "--model-version",
            "CostGrow_Terrain",
        ]
    )
    model_version, model_fp = _resolve_tohr_model_spec(args)
    assert model_version == "CostGrow_Terrain"
    assert model_fp is None


@pytest.mark.parametrize("case_id", [pytest.param("2407_FHIMP_tile", id="data_case_fetch_parse_2407_fhimp_tile")])
@pytest.mark.fast
def test_parse_tohr_allows_fetch_hrdem_without_dem(tile_case_d: dict):
    """Ensure tohr parser accepts --fetch-hrdem without requiring --dem."""
    case_spec = tile_case_d["case_spec"]
    parsed_args = _parse_arguments(
        [
            "tohr",
            "--in",
            str(tile_case_d["tile_dir"] / case_spec["inputs"]["lowres_fp"]),
            "--fetch-hrdem",
        ]
    )
    assert parsed_args.fetch_hrdem is True
    assert parsed_args.fetch_force_tiling is False
    assert parsed_args.dem is None


@pytest.mark.parametrize("case_id", [pytest.param("2407_FHIMP_tile", id="data_case_fetch_force_tiling_2407_fhimp_tile")])
@pytest.mark.fast
def test_parse_tohr_allows_fetch_force_tiling_flag(tile_case_d: dict):
    """Ensure tohr parser accepts --fetch-force-tiling when HRDEM fetch is enabled."""
    case_spec = tile_case_d["case_spec"]
    parsed_args = _parse_arguments(
        [
            "tohr",
            "--in",
            str(tile_case_d["tile_dir"] / case_spec["inputs"]["lowres_fp"]),
            "--fetch-hrdem",
            "--fetch-force-tiling",
        ]
    )
    assert parsed_args.fetch_hrdem is True
    assert parsed_args.fetch_force_tiling is True


@pytest.mark.parametrize("case_id", [pytest.param("2407_FHIMP_tile", id="data_case_min_depth_threshold_2407_fhimp_tile")])
@pytest.mark.fast
def test_parse_tohr_accepts_min_depth_threshold(tile_case_d: dict):
    """Ensure tohr parser accepts an explicit minimum retained depth threshold."""
    case_spec = tile_case_d["case_spec"]
    parsed_args = _parse_arguments(
        [
            "tohr",
            "--in",
            str(tile_case_d["tile_dir"] / case_spec["inputs"]["lowres_fp"]),
            "--fetch-hrdem",
            "--min-depth-threshold",
            "0.01",
        ]
    )
    assert parsed_args.fetch_hrdem is True
    assert parsed_args.min_depth_threshold == pytest.approx(0.01)


@pytest.mark.fast
def test_main_tohr_costgrow_builtin_runs_without_model_path(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_tohr_tiles: dict,
    capsys: pytest.CaptureFixture[str],
):
    """Ensure CLI ToHR can run the built-in CostGrow worker without a model artifact."""
    def fake_costgrow_core(
        pcraster_module,
        depth_lr,
        depth_lr_profile,
        dem_lr,
        dem_lr_valid_mask,
        dem_fine,
        dem_fine_valid_mask,
        fine_profile,
        min_depth_threshold,
        dp_coarse_pixel_max,
        decay_frac,
        distance_fill_method,
        distance_fill_kwargs,
    ):
        del pcraster_module, depth_lr, depth_lr_profile, dem_lr, dem_lr_valid_mask, fine_profile
        del min_depth_threshold, dp_coarse_pixel_max, decay_frac, distance_fill_method, distance_fill_kwargs
        pred = np.where(dem_fine_valid_mask, np.float32(0.25), np.nan).astype(np.float32, copy=False)
        return pred, {"wet_anchors": 1, "wet_final": int(np.isfinite(pred).sum())}

    monkeypatch.setattr(costgrow_module, "_check_pcraster", lambda: object())
    monkeypatch.setattr(costgrow_module, "_run_costgrow_core", fake_costgrow_core)
    monkeypatch.setattr(floodsr.tohr, "resolve_model_worker_class", lambda _: costgrow_module.ModelWorker)

    exit_code = main(
        [
            "tohr",
            "--in",
            str(synthetic_tohr_tiles["depth_lr_fp"]),
            "--dem",
            str(synthetic_tohr_tiles["dem_fp"]),
            "--out",
            str(synthetic_tohr_tiles["output_fp"]),
            "--model-version",
            "CostGrow_Terrain",
            "--window-method",
            "hard",
            "--tile-overlap",
            "0",
            "--no-progress",
        ]
    )
    output_fp = Path(capsys.readouterr().out.strip())
    assert exit_code == 0
    assert output_fp.exists()


@pytest.mark.fast
def test_main_tohr_costgrow_reports_missing_pcraster_error(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_tohr_tiles: dict,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    """Ensure CLI ToHR surfaces a clear error when CostGrow runtime deps are unavailable."""
    monkeypatch.setattr(costgrow_module, "_check_pcraster", lambda: (_ for _ in ()).throw(ImportError("PCRaster is required for CostGrow_Terrain")))
    monkeypatch.setattr(floodsr.tohr, "resolve_model_worker_class", lambda _: costgrow_module.ModelWorker)
    caplog.set_level("ERROR")

    exit_code = main(
        [
            "tohr",
            "--in",
            str(synthetic_tohr_tiles["depth_lr_fp"]),
            "--dem",
            str(synthetic_tohr_tiles["dem_fp"]),
            "--out",
            str(synthetic_tohr_tiles["output_fp"]),
            "--model-version",
            "CostGrow_Terrain",
            "--no-progress",
        ]
    )
    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert ("PCRaster is required for CostGrow_Terrain" in stderr) or ("PCRaster is required for CostGrow_Terrain" in caplog.text)


@pytest.mark.parametrize(
    ("progress_flag", "expected_show_progress"),
    [
        pytest.param("--show-progress", True, id="explicit_show_progress"),
        pytest.param("--no-progress", False, id="explicit_no_progress"),
    ],
)
@pytest.mark.parametrize("case_id", [pytest.param("2407_FHIMP_tile", id="data_case_progress_flag_2407_fhimp_tile")])
@pytest.mark.fast
def test_parse_tohr_accepts_progress_flags(tile_case_d: dict, progress_flag: str, expected_show_progress: bool):
    """Ensure tohr parser accepts the documented positive and negative progress flags."""
    case_spec = tile_case_d["case_spec"]
    parsed_args = _parse_arguments(
        [
            "tohr",
            "--in",
            str(tile_case_d["tile_dir"] / case_spec["inputs"]["lowres_fp"]),
            "--fetch-hrdem",
            progress_flag,
        ]
    )
    assert isinstance(parsed_args.show_progress, bool)
    assert parsed_args.show_progress is expected_show_progress


@pytest.mark.parametrize("case_id", [pytest.param("2407_FHIMP_tile", id="data_case_machine_json_2407_fhimp_tile")])
@pytest.mark.fast
def test_parse_tohr_allows_machine_json_only(tile_case_d: dict, tmp_path: Path):
    """Ensure tohr parser accepts machine-interface JSON as an alternate required-arg source."""
    case_spec = tile_case_d["case_spec"]
    machine_payload = {
        "in_fp": str(tile_case_d["tile_dir"] / case_spec["inputs"]["lowres_fp"]),
        "dem": str(tile_case_d["tile_dir"] / case_spec["inputs"]["dem_fp"]),
    }
    machine_json_fp = tmp_path / "tohr_machine.json"
    machine_json_fp.write_text(json.dumps(machine_payload), encoding="utf-8")

    parsed_args = _parse_arguments(["tohr", "--machine-json", str(machine_json_fp)])
    assert parsed_args.in_fp == Path(machine_payload["in_fp"])
    assert parsed_args.dem == Path(machine_payload["dem"])


@pytest.mark.parametrize("case_id", [pytest.param("2407_FHIMP_tile", id="data_case_machine_json_override_2407_fhimp_tile")])
@pytest.mark.fast
def test_parse_tohr_cli_args_override_machine_json(tile_case_d: dict, tmp_path: Path):
    """Ensure explicit CLI args retain precedence over machine-interface JSON."""
    case_spec = tile_case_d["case_spec"]
    machine_payload = {
        "in_fp": str(tile_case_d["tile_dir"] / case_spec["inputs"]["lowres_fp"]),
        "dem": str(tile_case_d["tile_dir"] / case_spec["inputs"]["dem_fp"]),
    }
    machine_json_fp = tmp_path / "tohr_machine_override.json"
    machine_json_fp.write_text(json.dumps(machine_payload), encoding="utf-8")
    override_input_fp = tmp_path / "override_input.tif"
    override_dem_fp = tmp_path / "override_dem.tif"

    parsed_args = _parse_arguments(
        [
            "tohr",
            "--machine-json",
            str(machine_json_fp),
            "--in",
            str(override_input_fp),
            "--dem",
            str(override_dem_fp),
        ]
    )
    assert parsed_args.in_fp == override_input_fp
    assert parsed_args.dem == override_dem_fp


@pytest.mark.parametrize("case_id", [pytest.param("rss_dudelange_A", id="data_case_min_depth_zeroes_output_rss_dudelange_a")])
@pytest.mark.e2e
def test_main_tohr_honors_min_depth_threshold(
    tohr_model_fp: Path,
    tmp_path: Path,
    tile_case_d: dict,
):
    """Ensure a high minimum depth threshold masks all predicted depths to zero."""
    pytest.importorskip("onnxruntime")
    rasterio = pytest.importorskip("rasterio")
    case_spec = tile_case_d["case_spec"]
    tile_dir = tile_case_d["tile_dir"]
    output_fp = tmp_path / f"{tile_case_d['case_name']}_pred_cli_min_depth.tif"

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
            str(tohr_model_fp),
            "--min-depth-threshold",
            "10.0",
        ]
    )
    with rasterio.open(output_fp) as ds:
        pred = ds.read(1)

    assert exit_code == 0
    assert np.count_nonzero(pred) == 0


@pytest.mark.parametrize("case_id", [pytest.param("2407_FHIMP_tile", id="data_case_dem_and_fetch_hrdem_2407_fhimp_tile")])
@pytest.mark.fast
def test_parse_tohr_rejects_dem_and_fetch_hrdem_together(tile_case_d: dict):
    """Ensure tohr parser rejects simultaneous --dem and --fetch-hrdem."""
    case_spec = tile_case_d["case_spec"]
    with pytest.raises(SystemExit):
        _parse_arguments(
            [
                "tohr",
                "--in",
                str(tile_case_d["tile_dir"] / case_spec["inputs"]["lowres_fp"]),
                "--dem",
                str(tile_case_d["tile_dir"] / case_spec["inputs"]["dem_fp"]),
                "--fetch-hrdem",
            ]
        )


@pytest.mark.parametrize("case_id", [pytest.param("2407_FHIMP_tile", id="data_case_fetch_out_requires_fetch_hrdem_2407_fhimp_tile")])
@pytest.mark.fast
def test_main_tohr_fetch_out_requires_fetch_hrdem(tile_case_d: dict, tmp_path: Path):
    """Ensure tohr runtime rejects --fetch-out unless --fetch-hrdem is enabled."""
    case_spec = tile_case_d["case_spec"]
    exit_code = main(
        [
            "tohr",
            "--in",
            str(tile_case_d["tile_dir"] / case_spec["inputs"]["lowres_fp"]),
            "--dem",
            str(tile_case_d["tile_dir"] / case_spec["inputs"]["dem_fp"]),
            "--fetch-out",
            str(tmp_path / "fetched_dem.tif"),
        ]
    )
    assert isinstance(exit_code, int)
    assert exit_code == 1
