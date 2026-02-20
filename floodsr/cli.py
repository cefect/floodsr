"""Command line interface for model operations."""

import argparse
import logging
from pathlib import Path

from floodsr.engine import get_onnxruntime_info, get_rasterio_info
from floodsr.inference import infer_geotiff
from floodsr.model_registry import fetch_model, list_models, load_models_manifest
from floodsr.cache_paths import get_model_cache_path
from floodsr.checksums import verify_sha256


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


def _resolve_infer_model_path(args: argparse.Namespace) -> Path:
    """Resolve infer model path from explicit file path or manifest version."""
    if args.model_path is not None:
        model_fp = Path(args.model_path).expanduser().resolve()
        assert model_fp.exists(), f"model path does not exist: {model_fp}"
        return model_fp
    if args.model_version is None:
        models = load_models_manifest(manifest_fp=args.manifest)
        assert models, "manifest has no model entries"

        first_version = next(iter(models))
        first_payload = models[first_version]
        first_fp = get_model_cache_path(first_version, first_payload["file_name"], cache_dir=args.cache_dir)
        if first_fp.exists() and verify_sha256(first_fp, first_payload["sha256"]):
            return first_fp

        for version, payload in models.items():
            cached_fp = get_model_cache_path(version, payload["file_name"], cache_dir=args.cache_dir)
            if cached_fp.exists() and verify_sha256(cached_fp, payload["sha256"]):
                return cached_fp

        raise FileNotFoundError(
            "no cached model found and --model-version was not provided. "
            "run `floodsr models fetch <model_version>` or pass --model-path."
        )
    return fetch_model(
        args.model_version,
        cache_dir=args.cache_dir,
        manifest_fp=args.manifest,
        backend_name=args.backend,
        force=args.force,
    )


def _resolve_default_output_path(in_fp: Path) -> Path:
    """Resolve default output in cwd from input filename."""
    in_path = Path(in_fp).expanduser()
    return (Path.cwd() / f"{in_path.stem}_sr.tif").resolve()


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

    # Route infer command.
    if args.command == "infer":
        model_fp = _resolve_infer_model_path(args)
        output_fp = args.out if args.out is not None else _resolve_default_output_path(args.in_fp)
        result = infer_geotiff(
            model_fp=model_fp,
            depth_lr_fp=args.in_fp,
            dem_hr_fp=args.dem,
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

    # Register inference command.
    infer_parser = subparsers.add_parser("infer", help="Run one GeoTIFF inference pass.")
    infer_parser.add_argument("--in", dest="in_fp", type=Path, required=True, help="Low-res depth raster path.")
    infer_parser.add_argument("--dem", type=Path, required=True, help="High-res DEM raster path.")
    infer_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output high-res depth raster path. Defaults to ./<input_stem>_sr.tif",
    )
    infer_parser.add_argument(
        "--model-version",
        default=None,
        help="Model version key from manifest when --model-path is not provided.",
    )
    infer_parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Explicit local ONNX model path.",
    )
    infer_parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to an alternate models.json manifest.",
    )
    infer_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for downloaded weights.",
    )
    infer_parser.add_argument(
        "--backend",
        choices=("http", "file"),
        default=None,
        help="Override retrieval backend selection for model fetch.",
    )
    infer_parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload when fetching a versioned model.",
    )
    infer_parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Optional max depth override for log-space scaling.",
    )
    infer_parser.add_argument(
        "--dem-pct-clip",
        type=float,
        default=None,
        help="Optional DEM percentile clip override when train stats are incomplete.",
    )
    infer_parser.add_argument(
        "--window-method",
        choices=("hard", "feather"),
        default="feather",
        help="Tile mosaicing method for inference.",
    )
    infer_parser.add_argument(
        "--tile-overlap",
        type=int,
        default=None,
        help="Feather overlap in low-res pixels. Ignored unless --window-method=feather.",
    )
    infer_parser.add_argument(
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
