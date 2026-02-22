"""CLI-friendly preprocess entrypoint for platform-model boundary preparation."""

import argparse, json, logging
from pathlib import Path

from floodsr.preprocessing import write_prepared_rasters


def main_preprocess(
    depth_lr_fp: str | Path,
    dem_hr_fp: str | Path,
    *,
    out_dir: str | Path,
    scale: int,
    depth_lr_prepared_fp: str | Path | None = None,
    dem_hr_prepared_fp: str | Path | None = None,
    logger=None,
) -> dict[str, object]:
    """
    Write platform-model boundary rasters used by model workers.

    Parameters
    ----------
    depth_lr_fp:
        Path to low-resolution depth raster.
    dem_hr_fp:
        Path to high-resolution DEM raster.
    out_dir:
        Directory where prepared artifacts are written.
    scale:
        Integer model scale factor.
    depth_lr_prepared_fp:
        Optional override output path for prepared low-resolution depth.
    dem_hr_prepared_fp:
        Optional override output path for prepared high-resolution DEM.
    logger:
        Optional logger instance.

    Returns
    -------
    dict[str, object]
        Prepared artifact metadata returned by `write_prepared_rasters`.
    """
    log = logger or logging.getLogger(__name__)
    assert int(scale) > 0, f"scale must be > 0; got {scale}"
    log.info(
        f"preprocess inputs\n"
        f"depth_lr_fp\n    {Path(depth_lr_fp).expanduser().resolve()}\n"
        f"dem_hr_fp\n    {Path(dem_hr_fp).expanduser().resolve()}\n"
        f"out_dir\n    {Path(out_dir).expanduser().resolve()}\n"
        f"scale={int(scale)}"
    )
    prepared = write_prepared_rasters(
        depth_lr_fp=depth_lr_fp,
        dem_hr_fp=dem_hr_fp,
        scale=int(scale),
        out_dir=out_dir,
        logger=log,
        depth_lr_prepared_fp=depth_lr_prepared_fp,
        dem_hr_prepared_fp=dem_hr_prepared_fp,
    )
    log.info(
        "preprocess outputs\n"
        f"depth_lr_prepared_fp\n    {prepared['depth_lr_prepared_fp']}\n"
        f"dem_hr_prepared_fp\n    {prepared['dem_hr_prepared_fp']}"
    )
    return prepared


def main(argv: list[str] | None = None) -> int:
    """Run preprocess CLI and print prepared artifact metadata JSON."""
    args = _parse_arguments(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    prepared = main_preprocess(
        args.depth_lr_fp,
        args.dem_hr_fp,
        out_dir=args.out_dir,
        scale=args.scale,
        depth_lr_prepared_fp=args.depth_lr_prepared_fp,
        dem_hr_prepared_fp=args.dem_hr_prepared_fp,
    )
    print(json.dumps({k: str(v) for k, v in prepared.items()}, indent=2, sort_keys=True))
    return 0


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for preprocess entrypoint."""
    parser = argparse.ArgumentParser(prog="python -m floodsr.preprocess", description="Prepare FloodSR boundary rasters.")
    parser.add_argument("--depth-lr-fp", type=Path, required=True, help="Low-resolution depth raster path.")
    parser.add_argument("--dem-hr-fp", type=Path, required=True, help="High-resolution DEM raster path.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for prepared rasters.")
    parser.add_argument("--scale", type=int, required=True, help="Integer model scale factor.")
    parser.add_argument(
        "--depth-lr-prepared-fp",
        type=Path,
        default=None,
        help="Optional output path override for prepared low-resolution depth raster.",
    )
    parser.add_argument(
        "--dem-hr-prepared-fp",
        type=Path,
        default=None,
        help="Optional output path override for prepared high-resolution DEM raster.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Stdlib logging level.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
