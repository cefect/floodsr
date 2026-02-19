"""Command line interface for model operations."""

import argparse
import logging
from pathlib import Path

from floodsr.model_registry import fetch_model, list_models


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
    logging.basicConfig(level=effective_level, force=True)


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

    raise ValueError(f"unsupported command path: {args.command}/{args.models_command}")


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
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
