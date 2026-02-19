"""Command line interface for model operations."""

import argparse
import sys
from pathlib import Path

from floodsr.model_registry import fetch_model, list_models


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
    try:
        return main_cli(args)
    except Exception as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for floodsr."""
    parser = argparse.ArgumentParser(prog="floodsr", description="FloodSR command line interface.")
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
