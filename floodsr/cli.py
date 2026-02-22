"""Command line interface for FloodSR operations."""

import argparse, logging
from pathlib import Path

from floodsr.cache_paths import get_model_cache_path
from floodsr.checksums import verify_sha256
from floodsr.dem_sources import fetch_dem
from floodsr.engine import get_onnxruntime_info, get_rasterio_info
from floodsr.model_registry import (
    fetch_model,
    list_models,
    list_runnable_model_versions,
    load_models_manifest,
    model_worker_exists,
)
from floodsr.tohr import tohr


log = logging.getLogger(__name__)


def _resolve_log_level(args: argparse.Namespace) -> int:
    """Resolve effective logging level from explicit level or verbosity flags."""
    if args.log_level is not None:
        return getattr(logging, args.log_level)

    # Start from INFO, then apply -v and -q offsets with DEBUG/ERROR clamp.
    level = logging.INFO - (10 * int(args.verbose)) + (10 * int(args.quiet))
    return max(logging.DEBUG, min(logging.ERROR, level))


def _configure_logging(args: argparse.Namespace) -> None:
    """Configure stdlib logging using Python default handler routing."""
    effective_level = _resolve_log_level(args)
    root_logger = logging.getLogger()
    root_logger.setLevel(effective_level)
    if not root_logger.handlers:
        logging.basicConfig(level=effective_level)


def _resolve_tohr_model_spec(args: argparse.Namespace) -> tuple[str, Path]:
    """Resolve ToHR model version/path from explicit file path or manifest/cache policy."""
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
    )


def _resolve_default_output_path(in_fp: Path) -> Path:
    """Resolve default output in cwd from input filename."""
    in_path = Path(in_fp).expanduser()
    suffix = in_path.suffix or ".tif"
    return (Path.cwd() / f"{in_path.stem}_sr{suffix}").resolve()


def main_cli(args: argparse.Namespace) -> int:
    """Run the CLI command selected by parsed arguments."""
    # Route model list command.
    if args.command == "models" and args.models_command == "list":
        for model in list_models(manifest_fp=args.manifest):
            print(f"{model.version}\t{model.file_name}\t{model.url}")
        return 0

    # Route model fetch command.
    if args.command == "models" and args.models_command == "fetch":
        model_fp = fetch_model(
            args.version,
            cache_dir=args.cache_dir,
            manifest_fp=args.manifest,
            backend_name=args.backend,
            force=args.force,
        )
        print(model_fp)
        return 0

    # Route main ToHR command.
    if args.command == "tohr":
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
                logger=log,
            )
            dem_fp = fetch_result.dem_fp

        result = tohr(
            model_version=model_version,
            model_fp=model_fp,
            depth_lr_fp=args.in_fp,
            dem_hr_fp=dem_fp,
            output_fp=output_fp,
            max_depth=args.max_depth,
            dem_pct_clip=args.dem_pct_clip,
            window_method=args.window_method,
            tile_overlap=args.tile_overlap,
            tile_size=args.tile_size,
            logger=log,
        )
        print(result["output_fp"])
        return 0

    # Route doctor command.
    if args.command == "doctor":
        ort_info = get_onnxruntime_info()
        rasterio_info = get_rasterio_info()
        print(f"onnxruntime_installed={ort_info['installed']}")
        print(f"onnxruntime_version={ort_info['version']}")
        print(f"onnxruntime_available_providers={','.join(ort_info['available_providers'])}")
        print(f"rasterio_installed={rasterio_info['installed']}")
        print(f"rasterio_version={rasterio_info['version']}")
        return 0

    raise ValueError(f"unsupported command path: {args.command}/{getattr(args, 'models_command', None)}")


def main(argv: list[str] | None = None) -> int:
    """Run the floodsr CLI and return an exit code."""
    args = _parse_arguments(argv)
    _configure_logging(args)
    try:
        return main_cli(args)
    except Exception as err:
        log.error(f"{err}")
        log.debug("unhandled CLI exception", exc_info=True)
        return 1


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for floodsr."""
    parser = argparse.ArgumentParser(prog="floodsr", description="FloodSR command line interface.")
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
    models_parser = subparsers.add_parser("models", help="Model registry commands.")
    models_subparsers = models_parser.add_subparsers(dest="models_command", required=True)

    models_list_parser = models_subparsers.add_parser("list", help="List available model versions.")
    models_list_parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to an alternate models.json manifest.",
    )

    models_fetch_parser = models_subparsers.add_parser("fetch", help="Fetch model weights by version.")
    models_fetch_parser.add_argument("version", help="Model version key from the manifest.")
    models_fetch_parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to an alternate models.json manifest.",
    )
    models_fetch_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for downloaded weights.",
    )
    models_fetch_parser.add_argument(
        "--backend",
        choices=("http", "file"),
        default=None,
        help="Override retrieval backend selection.",
    )
    models_fetch_parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even when a valid cache file exists.",
    )

    # Register ToHR command.
    tohr_parser = subparsers.add_parser("tohr", help="Run one raster ToHR pass.")
    tohr_parser.add_argument("--in", dest="in_fp", type=Path, required=True, help="Low-res depth raster path.")
    dem_group = tohr_parser.add_mutually_exclusive_group(required=True)
    dem_group.add_argument("--dem", type=Path, default=None, help="High-res DEM raster path.")
    dem_group.add_argument(
        "-f",
        "--fetch-hrdem",
        action="store_true",
        help="Fetch HRDEM from STAC using the low-res raster footprint.",
    )
    tohr_parser.add_argument(
        "--fetch-out",
        type=Path,
        default=None,
        help="Optional output path for fetched HRDEM tile. Defaults to temp directory.",
    )
    tohr_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output high-res depth raster path. Defaults to ./<input_stem>_sr with input extension",
    )
    tohr_parser.add_argument(
        "--model-version",
        default=None,
        help="Model version key from manifest when --model-path is not provided.",
    )
    tohr_parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Explicit local ONNX model path.",
    )
    tohr_parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to an alternate models.json manifest.",
    )
    tohr_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for downloaded weights.",
    )
    tohr_parser.add_argument(
        "--backend",
        choices=("http", "file"),
        default=None,
        help="Override retrieval backend selection for model fetch.",
    )
    tohr_parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload when fetching a versioned model.",
    )
    tohr_parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Optional max depth override for log-space scaling.",
    )
    tohr_parser.add_argument(
        "--dem-pct-clip",
        type=float,
        default=None,
        help="Optional DEM percentile clip override when train stats are incomplete.",
    )
    tohr_parser.add_argument(
        "--window-method",
        choices=("hard", "feather"),
        default="feather",
        help="Tile mosaicing method for ToHR.",
    )
    tohr_parser.add_argument(
        "--tile-overlap",
        type=int,
        default=None,
        help="Feather overlap in low-res pixels. Ignored unless --window-method=feather.",
    )
    tohr_parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="LR tile size override (must match model LR input size).",
    )

    # Register diagnostic command.
    subparsers.add_parser("doctor", help="Report runtime dependency diagnostics.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
