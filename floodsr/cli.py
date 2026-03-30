"""Command-line entrypoints for FloodSR user and machine interfaces."""

import argparse, json, logging, sys
from pathlib import Path

log = logging.getLogger(__name__)


def _resolve_log_level(args: argparse.Namespace) -> int:
    """Resolve the effective log level from `--log-level`, `-v`, and `-q`."""
    if args.log_level is not None:
        return getattr(logging, args.log_level)

    # Start from INFO, then apply -v and -q offsets with DEBUG/ERROR clamp.
    level = logging.INFO - (10 * int(args.verbose)) + (10 * int(args.quiet))
    return max(logging.DEBUG, min(logging.ERROR, level))


def _configure_logging(args: argparse.Namespace) -> None:
    """Configure root logging for the parsed CLI arguments."""
    effective_level = _resolve_log_level(args)
    root_logger = logging.getLogger()
    root_logger.setLevel(effective_level)
    if not root_logger.handlers:
        logging.basicConfig(level=effective_level)


def _resolve_tohr_model_spec(args: argparse.Namespace) -> tuple[str, Path]:
    """Resolve the model worker version and ONNX path for `floodsr tohr`."""
    from floodsr.cache_paths import get_model_cache_path
    from floodsr.checksums import verify_sha256
    from floodsr.model_registry import fetch_model, list_runnable_model_versions, load_models_manifest, model_worker_exists

    if args.model_path is not None:
        model_fp = Path(args.model_path).expanduser().resolve()
        assert model_fp.exists(), f"model path does not exist: {model_fp}"
        if args.model_version is not None:
            if not model_worker_exists(args.model_version):
                raise ValueError(f"no model worker found for --model-version={args.model_version}")
            return args.model_version, model_fp

        runnable_versions = list_runnable_model_versions(manifest_fp=args.manifest)
        assert runnable_versions, "manifest has no runnable model entries"
        return runnable_versions[0], model_fp

    models = load_models_manifest(manifest_fp=args.manifest)
    assert models, "manifest has no model entries"
    runnable_versions = [version for version in models if model_worker_exists(version)]
    assert runnable_versions, "manifest has no runnable model entries (worker module missing)"

    if args.model_version is None:
        # Try first listed runnable model first, then fallback to first valid cached runnable model.
        first_version = runnable_versions[0]
        first_payload = models[first_version]
        first_fp = get_model_cache_path(first_version, first_payload["file_name"], cache_dir=args.cache_dir)
        if first_fp.exists() and verify_sha256(first_fp, first_payload["sha256"]):
            return first_version, first_fp

        for version in runnable_versions:
            payload = models[version]
            cached_fp = get_model_cache_path(version, payload["file_name"], cache_dir=args.cache_dir)
            if cached_fp.exists() and verify_sha256(cached_fp, payload["sha256"]):
                return version, cached_fp

        raise FileNotFoundError(
            "no cached runnable model found and --model-version was not provided. "
            "run `floodsr models fetch <model_version>` or pass --model-path."
        )

    if not model_worker_exists(args.model_version):
        raise ValueError(f"no model worker found for --model-version={args.model_version}")
    return args.model_version, fetch_model(
        args.model_version,
        cache_dir=args.cache_dir,
        manifest_fp=args.manifest,
        backend_name=args.backend,
        force=args.force,
        show_progress=args.show_progress,
    )


def _find_flag_value(argv: list[str], flag: str) -> str | None:
    """Return one raw CLI flag value from `argv`."""
    for idx, token in enumerate(argv):
        if token == flag:
            return argv[idx + 1] if idx + 1 < len(argv) else None
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def _flag_present(argv: list[str], flag: str) -> bool:
    """Return whether one CLI flag is already present in `argv`."""
    return any(token == flag or token.startswith(f"{flag}=") for token in argv)


def _add_progress_arguments(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add shared positive/negative progress flags with an explicit default."""
    # Keep the positive flag as the canonical interface while offering a short negative alias.
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--show-progress",
        dest="show_progress",
        action="store_true",
        default=True,
        help=help_text,
    )
    progress_group.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Disable progress output. Default: progress is shown.",
    )


def _read_tohr_machine_json(machine_json_fp: Path) -> dict[str, object]:
    """Load a `tohr` machine-json payload from disk."""
    machine_json_path = machine_json_fp.expanduser().resolve()
    assert machine_json_path.exists(), f"machine json does not exist: {machine_json_path}"
    payload = json.loads(machine_json_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"machine json must be an object: {machine_json_path}"
    # Allow either a direct payload or a nested `tohr` payload.
    if "tohr" in payload:
        nested_payload = payload["tohr"]
        assert isinstance(nested_payload, dict), f"machine json 'tohr' payload must be an object: {machine_json_path}"
        return nested_payload
    return payload


def _normalize_machine_key(raw_key: str) -> str:
    """Normalize one machine-json key to an argparse destination name."""
    return raw_key.strip().lstrip("-").replace("-", "_")


def _build_tohr_machine_cli_tokens(payload: dict[str, object], argv: list[str]) -> list[str]:
    """Translate supported machine-json `tohr` keys into CLI tokens."""
    # Keep this mapping aligned with `_parse_arguments()` ToHR option destinations.
    machine_key_to_flag = {
        "in": "--in",
        "in_fp": "--in",
        "dem": "--dem",
        "fetch_hrdem": "--fetch-hrdem",
        "fetch_out": "--fetch-out",
        "fetch_force_tiling": "--fetch-force-tiling",
        "out": "--out",
        "model_version": "--model-version",
        "model_path": "--model-path",
        "manifest": "--manifest",
        "cache_dir": "--cache-dir",
        "backend": "--backend",
        "force": "--force",
        "max_depth": "--max-depth",
        "min_depth_threshold": "--min-depth-threshold",
        "dem_pct_clip": "--dem-pct-clip",
        "window_method": "--window-method",
        "tile_overlap": "--tile-overlap",
        "tile_size": "--tile-size",
        "crs_policy": "--crs-policy",
        "show_progress": "--show-progress",
    }
    bool_flags = {"fetch_hrdem", "fetch_force_tiling", "force", "show_progress"}
    cli_tokens = []
    for raw_key, value in payload.items():
        key = _normalize_machine_key(raw_key)
        if key not in machine_key_to_flag:
            raise ValueError(f"unsupported tohr machine-json key: {raw_key}")
        cli_flag = machine_key_to_flag[key]
        # Preserve explicit CLI args as highest precedence.
        if _flag_present(argv, cli_flag):
            continue
        if key in bool_flags:
            if not isinstance(value, bool):
                raise ValueError(f"machine-json key '{raw_key}' must be boolean, got {type(value)!r}")
            if key == "show_progress":
                cli_tokens.append(cli_flag if value else "--no-progress")
            elif value:
                cli_tokens.append(cli_flag)
            continue
        if value is None:
            continue
        cli_tokens.extend([cli_flag, str(value)])
    return cli_tokens


def _inject_tohr_machine_json_args(argv: list[str] | None) -> list[str] | None:
    """Expand `--machine-json` into parser-ready `tohr` CLI arguments."""
    if argv is None:
        argv_tokens = list(sys.argv[1:])
    else:
        argv_tokens = list(argv)
    if not argv_tokens or argv_tokens[0] != "tohr":
        return argv_tokens
    machine_json_raw = _find_flag_value(argv_tokens, "--machine-json")
    if machine_json_raw is None:
        return argv_tokens
    machine_payload = _read_tohr_machine_json(Path(machine_json_raw))
    return argv_tokens + _build_tohr_machine_cli_tokens(machine_payload, argv_tokens)


def _resolve_default_output_path(in_fp: Path) -> Path:
    """Build the default output path in the current working directory."""
    in_path = Path(in_fp).expanduser()
    suffix = in_path.suffix or ".tif"
    return (Path.cwd() / f"{in_path.stem}_sr{suffix}").resolve()


def _build_floodsr_package_info() -> dict[str, str]:
    """Collect installed FloodSR package metadata for CLI diagnostics."""
    import floodsr

    return {
        "version": floodsr.__version__,
        "module_path": str(Path(floodsr.__file__).resolve()),
    }


def _build_doctor_payload() -> dict[str, object]:
    """Collect runtime dependency diagnostics for the `doctor` command."""
    from floodsr.engine import get_gdal_info, get_onnxruntime_info, get_rasterio_info

    floodsr_info = _build_floodsr_package_info()
    ort_info = get_onnxruntime_info()
    rasterio_info = get_rasterio_info()
    gdal_info = get_gdal_info()
    return {
        "floodsr": floodsr_info,
        "onnxruntime": ort_info,
        "rasterio": rasterio_info,
        "gdal": gdal_info,
    }


def main_cli(args: argparse.Namespace) -> int:
    """Dispatch the parsed CLI command and return its exit status."""
    # Route model list command.
    if args.command == "models" and args.models_command == "list":
        from floodsr.model_registry import list_models

        for model in list_models(manifest_fp=args.manifest):
            print(f"{model.version}\t{model.file_name}\t{model.url}")
        return 0

    # Route model fetch command.
    if args.command == "models" and args.models_command == "fetch":
        from floodsr.model_registry import fetch_model

        model_fp = fetch_model(
            args.version,
            cache_dir=args.cache_dir,
            manifest_fp=args.manifest,
            backend_name=args.backend,
            force=args.force,
            show_progress=args.show_progress,
        )
        print(model_fp)
        return 0

    # Route main ToHR command.
    if args.command == "tohr":
        # Defer heavy raster imports until the ToHR path is actually used.
        from floodsr.dem_sources import fetch_dem
        from floodsr.tohr import tohr

        if args.fetch_out is not None and not args.fetch_hrdem:
            raise ValueError("--fetch-out requires --fetch-hrdem")

        model_version, model_fp = _resolve_tohr_model_spec(args)
        output_fp = args.out if args.out is not None else _resolve_default_output_path(args.in_fp)
        dem_fp = args.dem
        if args.fetch_hrdem:
            fetch_result = fetch_dem(
                source_id="hrdem",
                depth_lr_fp=args.in_fp,
                output_fp=args.fetch_out,
                cache_dir=args.cache_dir,
                fetch_force_tiling=args.fetch_force_tiling,
                show_progress=args.show_progress,
                logger=log,
            )
            dem_fp = fetch_result.dem_fp

        result = tohr(
            model_version=model_version,
            model_fp=model_fp,
            depth_lr_fp=args.in_fp,
            dem_hr_fp=dem_fp,
            output_fp=output_fp,
            crs_policy=args.crs_policy,
            max_depth=args.max_depth,
            min_depth_threshold=args.min_depth_threshold,
            dem_pct_clip=args.dem_pct_clip,
            window_method=args.window_method,
            tile_overlap=args.tile_overlap,
            tile_size=args.tile_size,
            show_progress=args.show_progress,
            logger=log,
        )
        print(result["output_fp"])
        return 0

    # Route doctor command.
    if args.command == "doctor":
        payload = _build_doctor_payload()
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        floodsr_info = payload["floodsr"]
        ort_info = payload["onnxruntime"]
        rasterio_info = payload["rasterio"]
        gdal_info = payload["gdal"]
        print(f"floodsr_version={floodsr_info['version']}")
        print(f"floodsr_module_path={floodsr_info['module_path']}")
        print(f"onnxruntime_installed={ort_info['installed']}")
        print(f"onnxruntime_version={ort_info['version']}")
        print(f"onnxruntime_available_providers={','.join(ort_info['available_providers'])}")
        print(f"rasterio_installed={rasterio_info['installed']}")
        print(f"rasterio_version={rasterio_info['version']}")
        print(f"gdal_python_installed={gdal_info['python_bindings_installed']}")
        print(f"gdal_python_version={gdal_info['python_bindings_version']}")
        print(f"gdal_config_installed={gdal_info['gdal_config_installed']}")
        print(f"gdal_config_version={gdal_info['gdal_config_version']}")
        print(f"gdal_vrt_enabled={gdal_info['vrt_enabled']}")
        return 0

    raise ValueError(f"unsupported command path: {args.command}/{getattr(args, 'models_command', None)}")


def main(argv: list[str] | None = None) -> int:
    """Run the FloodSR CLI and return a process exit code."""
    try:
        args = _parse_arguments(argv)
        _configure_logging(args)
        return main_cli(args)
    except SystemExit as err:
        return 0 if err.code in (None, 0) else int(err.code)
    except Exception as err:
        log.error(f"{err}")
        log.debug("unhandled CLI exception", exc_info=True)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the FloodSR CLI parser."""
    floodsr_info = _build_floodsr_package_info()
    parser = argparse.ArgumentParser(
        prog="floodsr",
        description="Run FloodSR model, cache, and runtime utility commands.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {floodsr_info['version']}",
        help="Print the installed FloodSR package version and exit.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (repeatable).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help="Decrease logging verbosity (repeatable).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=None,
        help="Explicit log level override.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register model-related commands.
    models_parser = subparsers.add_parser(
        "models",
        help="List manifest models or fetch cached model weights.",
        description="List manifest models or fetch cached model weights.",
    )
    models_subparsers = models_parser.add_subparsers(dest="models_command", required=True)

    models_list_parser = models_subparsers.add_parser(
        "list",
        help="List model versions defined in the manifest.",
        description="List model versions defined in the manifest.",
    )
    models_list_parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Read models from an alternate `models.json` manifest.",
    )

    models_fetch_parser = models_subparsers.add_parser(
        "fetch",
        help="Fetch one manifest model into the local cache.",
        description="Fetch one manifest model into the local cache.",
    )
    models_fetch_parser.add_argument("version", help="Model version key to fetch from the manifest.")
    models_fetch_parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Read models from an alternate `models.json` manifest.",
    )
    models_fetch_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Store downloaded weights in an alternate cache directory.",
    )
    models_fetch_parser.add_argument(
        "--backend",
        choices=("http", "file"),
        default=None,
        help="Override weight retrieval backend selection.",
    )
    models_fetch_parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even when a valid cached weight file already exists.",
    )
    _add_progress_arguments(
        models_fetch_parser,
        help_text="Show progress output during model download. Use `--no-progress` to disable it. Default: enabled.",
    )

    # Register ToHR command.
    tohr_parser = subparsers.add_parser(
        "tohr",
        help="Run one super-resolution pass for a low-res depth raster.",
        description="Run one super-resolution pass for a low-res depth raster.",
    )
    tohr_parser.add_argument(
        "--machine-json",
        type=Path,
        default=None,
        help="Load `tohr` parameters from JSON; explicit CLI flags still take precedence.",
    )
    tohr_parser.add_argument("--in", dest="in_fp", type=Path, required=True, help="Input low-res depth raster path.")
    dem_group = tohr_parser.add_mutually_exclusive_group(required=True)
    dem_group.add_argument("--dem", type=Path, default=None, help="Input high-res DEM raster path.")
    dem_group.add_argument(
        "-f",
        "--fetch-hrdem",
        action="store_true",
        help="Fetch HRDEM for the low-res raster footprint instead of passing `--dem`.",
    )
    tohr_parser.add_argument(
        "--fetch-out",
        type=Path,
        default=None,
        help="Write a fetched HRDEM raster to this path instead of a temporary location.",
    )
    tohr_parser.add_argument(
        "--fetch-force-tiling",
        action="store_true",
        help="Force tiled HRDEM fetch windows instead of relying on automatic tiling.",
    )
    tohr_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output raster path. Defaults to `./<input_stem>_sr<input_suffix>` in the current working directory.",
    )
    tohr_parser.add_argument(
        "--model-version",
        default=None,
        help="Manifest model version to run or fetch when `--model-path` is not provided.",
    )
    tohr_parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Use an explicit local ONNX model file instead of resolving from cache/manifest.",
    )
    tohr_parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Read model metadata from an alternate `models.json` manifest.",
    )
    tohr_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Use an alternate cache directory for resolved model weights.",
    )
    tohr_parser.add_argument(
        "--backend",
        choices=("http", "file"),
        default=None,
        help="Override model weight retrieval backend selection.",
    )
    tohr_parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload a versioned model when cache resolution would otherwise reuse it.",
    )
    tohr_parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help=(
            "Maximum depth used when clipping low-res depth values before log1p scaling into the model input range. "
            "Values above this threshold are capped during preprocessing and after inverse scaling. "
            "The current ResUNet_16x_DEM default resolves to 5.0."
        ),
    )
    tohr_parser.add_argument(
        "--min-depth-threshold",
        type=float,
        default=None,
        help=(
            "Minimum predicted depth retained in the final raster. "
            "Values below this threshold are written as 0.0 after inference. "
            "The current ResUNet_16x_DEM default resolves to 0.01."
        ),
    )
    tohr_parser.add_argument(
        "--dem-pct-clip",
        type=float,
        default=None,
        help=(
            "Percentile used to cap high DEM values before min-max normalization to [0, 1]. "
            "Lower values clip high terrain more aggressively; higher values preserve more of the upper tail. "
            "Used when explicit DEM normalization stats are unavailable. "
            "The current ResUNet_16x_DEM default resolves to 95.0."
        ),
    )
    tohr_parser.add_argument(
        "--window-method",
        choices=("hard", "feather"),
        default="feather",
        help=(
            "Tile mosaicing method used when stitching model windows. "
            "`hard` uses non-overlapping tiles with direct writes; "
            "`feather` uses overlapping tiles with weighted blending to reduce seam artifacts. "
            "The current default is `feather`."
        ),
    )
    tohr_parser.add_argument(
        "--tile-overlap",
        type=int,
        default=None,
        help="Feather overlap in low-res pixels. Ignored unless `--window-method=feather`.",
    )
    tohr_parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Override the low-res tile size; must match the model LR input size.",
    )
    tohr_parser.add_argument(
        "--crs-policy",
        choices=("strict", "use-dem", "use-lores"),
        default="strict",
        help="Policy for CRS mismatches between the low-res depth raster and DEM.",
    )
    _add_progress_arguments(
        tohr_parser,
        help_text=(
            "Show progress output during model download, DEM fetch, and tiled runtime work. "
            "Use `--no-progress` to disable it. Default: enabled."
        ),
    )

    # Register diagnostic command.
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Report runtime dependency and provider diagnostics.",
        description="Report runtime dependency and provider diagnostics.",
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of line-oriented text.",
    )
    return parser


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments after optional `tohr --machine-json` expansion."""
    return build_parser().parse_args(_inject_tohr_machine_json_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
