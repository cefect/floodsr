"""Inference execution for FloodSR raster workflows."""

import logging
import math
import tempfile
import time
from pathlib import Path
from typing import Dict

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    tqdm = None

from floodsr.preprocessing import (
    _build_tile_starts,
    normalize_dem,
    _read_single_band_raster,
    _write_single_band_raster,
    scale_depth_log1p_np,
    resolve_preprocess_config,
    write_prepared_rasters,
)


def compute_depth_error_metrics(
    reference_depth_m: np.ndarray,
    estimate_depth_m: np.ndarray,
    max_depth: float,
    dry_depth_thresh_m: float = 1e-3,
) -> Dict[str, float]:
    """Compute pairwise depth error metrics for one tile."""
    assert reference_depth_m.ndim == 2, f"reference depth must be 2D; got {reference_depth_m.shape}"
    assert estimate_depth_m.shape == reference_depth_m.shape, (
        f"estimate shape {estimate_depth_m.shape} must match reference shape {reference_depth_m.shape}"
    )
    assert max_depth > 0, f"max_depth must be > 0; got {max_depth}"

    reference_np = reference_depth_m.astype(np.float32, copy=False)
    estimate_np = estimate_depth_m.astype(np.float32, copy=False)

    diff = estimate_np - reference_np
    wet_mask = reference_np >= dry_depth_thresh_m
    wet_pixel_count = int(wet_mask.sum())
    total_pixels = int(reference_np.size)
    dry_pixel_count = int(total_pixels - wet_pixel_count)

    mse_all = float(np.mean(np.square(diff), dtype=np.float64))
    rmse_all = float(np.sqrt(mse_all))
    mae_all = float(np.mean(np.abs(diff), dtype=np.float64))
    bias_all = float(np.mean(diff, dtype=np.float64))

    if wet_pixel_count > 0:
        rmse_wet = float(np.sqrt(np.mean(np.square(diff[wet_mask]), dtype=np.float64)))
    else:
        rmse_wet = np.nan

    if mse_all <= 0.0:
        psnr = np.inf
    else:
        psnr = float(20.0 * np.log10(max_depth) - 10.0 * np.log10(mse_all))

    ref64 = reference_np.astype(np.float64, copy=False)
    est64 = estimate_np.astype(np.float64, copy=False)
    mu_x = float(ref64.mean())
    mu_y = float(est64.mean())
    sigma_x = float(ref64.var())
    sigma_y = float(est64.var())
    sigma_xy = float(((ref64 - mu_x) * (est64 - mu_y)).mean())

    c1 = float((0.01 * max_depth) ** 2)
    c2 = float((0.03 * max_depth) ** 2)
    ssim_num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    ssim_den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    ssim = float(ssim_num / ssim_den) if ssim_den != 0.0 else np.nan

    return {
        "psnr": psnr,
        "ssim": ssim,
        "rmse_m": rmse_all,
        "rmse_wet_m": rmse_wet,
        "mae_m": mae_all,
        "mase_m": mae_all,
        "bias_m": bias_all,
        "mse_m2": mse_all,
        "dry_pixel_count": dry_pixel_count,
        "wet_pixel_count": wet_pixel_count,
    }


def _pixel_size_m(profile: dict) -> tuple[float, float]:
    """Extract absolute pixel size in projection units."""
    transform = profile.get("transform")
    if transform is None:
        return (float("nan"), float("nan"))
    if hasattr(transform, "a") and hasattr(transform, "e"):
        return (abs(float(transform.a)), abs(float(transform.e)))
    return (abs(float(transform[0])), abs(float(transform[4])))


def _iter_windows(y_starts: list[int], x_starts: list[int], use_progress: bool):
    """Build iterable of window origins with optional tqdm progress."""
    total = len(y_starts) * len(x_starts)
    windows = ((yi, xi, y0, x0) for yi, y0 in enumerate(y_starts) for xi, x0 in enumerate(x_starts))
    if use_progress and tqdm is not None:
        return tqdm(windows, desc="windowed inference", total=total, unit="window")
    return windows


def infer_from_prepared_inputs(
    *,
    engine,
    depth_lr_fp: str | Path,
    dem_hr_fp: str | Path,
    preprocess_cfg: Dict[str, object],
    model_lr_tile: int,
    model_scale: int,
    contract_hr_tile: int,
    window_method: str,
    overlap_lr: int,
    logger=None,
) -> tuple[np.ndarray, int]:
    """Run tiled inference against prepared depth/DEM rasters."""
    log = logger or logging.getLogger(__name__)
    assert window_method in {"hard", "feather"}, f"unsupported window_method={window_method}"

    depth_lr_raw, _depth_lr_nodata, depth_lr_profile = _read_single_band_raster(depth_lr_fp)
    dem_hr_raw, _dem_hr_nodata, dem_hr_profile = _read_single_band_raster(dem_hr_fp)
    assert depth_lr_raw.ndim == 2, f"aligned depth must be 2D; got {depth_lr_raw.shape}"
    assert dem_hr_raw.ndim == 2, f"aligned DEM must be 2D; got {dem_hr_raw.shape}"

    max_depth = float(preprocess_cfg["max_depth"])
    dem_pct_clip = float(preprocess_cfg["dem_pct_clip"])
    dem_ref_stats = preprocess_cfg["dem_ref_stats"]

    depth_lr_norm = scale_depth_log1p_np(depth_lr_raw, max_depth=max_depth)
    dem_hr_norm, dem_stats_used = normalize_dem(dem_hr_raw, pct_clip=dem_pct_clip, ref_stats=dem_ref_stats)
    assert depth_lr_norm is not None, "depth normalization returned None"
    assert dem_hr_norm is not None, "DEM normalization returned None"
    assert dem_stats_used is not None, "DEM normalization did not return stats"
    assert np.isfinite(depth_lr_norm).all(), "normalized low-res depth contains non-finite values"
    assert np.isfinite(dem_hr_norm).all(), "normalized DEM contains non-finite values"
    log.info(
        "input/preprocess summary:\n"
        f"  aligned depth_lr shape={depth_lr_raw.shape} res={_pixel_size_m(depth_lr_profile)} m/pix\n"
        f"  aligned dem_hr shape={dem_hr_raw.shape} res={_pixel_size_m(dem_hr_profile)} m/pix\n"
        f"  max_depth={max_depth}\n"
        f"  dem_pct_clip={dem_pct_clip}\n"
        f"  dem_ref_stats={dem_stats_used}\n"
        f"  normalized depth range={float(np.nanmin(depth_lr_norm)):.6f}..{float(np.nanmax(depth_lr_norm)):.6f}\n"
        f"  normalized dem range={float(np.nanmin(dem_hr_norm)):.6f}..{float(np.nanmax(dem_hr_norm)):.6f}"
    )

    crop_h, crop_w = dem_hr_norm.shape
    assert crop_h > 0 and crop_w > 0, f"aligned DEM has invalid shape {(crop_h, crop_w)}"
    expected_lr_h = crop_h // model_scale
    expected_lr_w = crop_w // model_scale
    assert expected_lr_h > 0 and expected_lr_w > 0, (
        f"expected low-resolution shape invalid {(expected_lr_h, expected_lr_w)} from crop {(crop_h, crop_w)} and scale={model_scale}"
    )
    assert depth_lr_norm.shape == (expected_lr_h, expected_lr_w), (
        f"depth shape {depth_lr_raw.shape} does not match crop/scale target {(expected_lr_h, expected_lr_w)}"
    )

    if depth_lr_raw.min() > max_depth:
        log.warning("low-res depth values exceed max_depth; model preprocessing will clip them.")

    pad_h = (int(math.ceil(crop_h / contract_hr_tile)) * contract_hr_tile) - crop_h
    pad_w = (int(math.ceil(crop_w / contract_hr_tile)) * contract_hr_tile) - crop_w

    dem_pad = np.pad(
        dem_hr_norm,
        ((0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=0.0,
    )
    depth_pad = np.pad(
        depth_lr_norm,
        ((0, pad_h // model_scale), (0, pad_w // model_scale)),
        mode="constant",
        constant_values=0.0,
    )

    hr_pad_h, hr_pad_w = dem_pad.shape
    depth_pad_h, depth_pad_w = depth_pad.shape
    assert depth_pad_h == hr_pad_h // model_scale and depth_pad_w == hr_pad_w // model_scale, (
        f"depth pad shape {(depth_pad_h, depth_pad_w)} incompatible with HR pad {(hr_pad_h, hr_pad_w)}"
    )

    overlap_hr = overlap_lr * model_scale
    tile_cache: dict[tuple[int, int], np.ndarray] = {}

    log.info(
        "window config\n"
        f"  method={window_method}\n"
        f"  overlap_lr={overlap_lr}\n"
        f"  overlap_hr={overlap_hr}\n"
        f"  tile_size_lr={model_lr_tile}\n"
        f"  tile_size_hr={contract_hr_tile}"
    )

    def _predict_tile(y0: int, x0: int) -> np.ndarray:
        key = (int(y0), int(x0))
        if key in tile_cache:
            return tile_cache[key]

        lr_y0 = y0 // model_scale
        lr_x0 = x0 // model_scale
        depth_tile = depth_pad[
            lr_y0 : lr_y0 + model_lr_tile,
            lr_x0 : lr_x0 + model_lr_tile,
        ]
        dem_tile = dem_pad[y0 : y0 + contract_hr_tile, x0 : x0 + contract_hr_tile]
        assert depth_tile.shape == (model_lr_tile, model_lr_tile), (
            f"depth tile shape {depth_tile.shape} != {(model_lr_tile, model_lr_tile)}"
        )
        assert dem_tile.shape == (contract_hr_tile, contract_hr_tile), (
            f"DEM tile shape {dem_tile.shape} != {(contract_hr_tile, contract_hr_tile)}"
        )
        run_result = engine.run_tile(
            depth_tile,
            dem_tile,
            max_depth=max_depth,
            dem_pct_clip=dem_pct_clip,
            dem_ref_stats=dem_ref_stats,
            normalize_inputs=False,
            depth_lr_nodata=None,
            dem_hr_nodata=None,
            logger=log,
        )
        pred = run_result["prediction_m"]
        assert pred.shape == (contract_hr_tile, contract_hr_tile), (
            f"prediction shape {pred.shape} != {(contract_hr_tile, contract_hr_tile)}"
        )
        tile_cache[key] = pred
        return pred

    # Prime tile cache with a non-overlap pass (same as notebook inference path).
    nonoverlap_y = list(range(0, hr_pad_h, contract_hr_tile))
    nonoverlap_x = list(range(0, hr_pad_w, contract_hr_tile))
    sr_pad_nonoverlap = np.zeros_like(dem_pad, dtype=np.float32)
    log.info(
        f"priming tile cache with non-overlap pass over {len(nonoverlap_y) * len(nonoverlap_x)} windows\n"
        f"  overlap_lr={overlap_lr}\n"
        f"  overlap_hr={overlap_hr}"
    )
    for _, _, y0, x0 in _iter_windows(nonoverlap_y, nonoverlap_x, use_progress=True):
        pred_np = _predict_tile(y0, x0)
        sr_pad_nonoverlap[y0 : y0 + contract_hr_tile, x0 : x0 + contract_hr_tile] = pred_np

    if window_method == "hard":
        sr_pad = sr_pad_nonoverlap.copy()
    elif window_method == "feather":
        stride_hr = contract_hr_tile - overlap_hr
        if overlap_lr <= 0:
            raise AssertionError("feather windowing requires overlap_lr > 0")
        if stride_hr <= 0:
            raise AssertionError(
                f"feather stride must be > 0; overlap_lr={overlap_lr}, tile={contract_hr_tile}"
            )

        y_starts = _build_tile_starts(hr_pad_h, contract_hr_tile, stride_hr)
        x_starts = _build_tile_starts(hr_pad_w, contract_hr_tile, stride_hr)
        feather_1d = np.ones(contract_hr_tile, dtype=np.float32)
        if overlap_hr > 0:
            ramp = np.linspace(0.0, 1.0, overlap_hr + 2, dtype=np.float32)[1:-1]
            feather_1d[:overlap_hr] = ramp
            feather_1d[-overlap_hr:] = ramp[::-1]
            feather_1d = np.clip(feather_1d, 1e-3, 1.0)

        accum = np.zeros_like(dem_pad, dtype=np.float32)
        weight_sum = np.zeros_like(dem_pad, dtype=np.float32)
        log.info(
            f"running feather tiling over {len(y_starts)}x{len(x_starts)} grid\n"
            f"  stride_hr={stride_hr}\n"
            f"  overlap_hr={overlap_hr}"
        )
        for yi, xi, y0, x0 in _iter_windows(y_starts, x_starts, use_progress=True):
            pred_np = _predict_tile(y0, x0)
            wy = feather_1d.copy()
            wx = feather_1d.copy()
            if yi == 0:
                wy[:overlap_hr] = 1.0
            if yi == len(y_starts) - 1:
                wy[-overlap_hr:] = 1.0
            if xi == 0:
                wx[:overlap_hr] = 1.0
            if xi == len(x_starts) - 1:
                wx[-overlap_hr:] = 1.0

            weight = np.outer(wy, wx).astype(np.float32, copy=False)
            accum[y0 : y0 + contract_hr_tile, x0 : x0 + contract_hr_tile] += pred_np * weight
            weight_sum[y0 : y0 + contract_hr_tile, x0 : x0 + contract_hr_tile] += weight

        sr_pad = np.divide(
            accum,
            np.maximum(weight_sum, 1e-6),
            out=np.zeros_like(accum),
            where=weight_sum > 0,
        )
    else:
        raise AssertionError(f"unsupported window_method={window_method}")

    prediction_m = np.clip(sr_pad[:crop_h, :crop_w], 0.0, float(preprocess_cfg["max_depth"]))
    assert prediction_m.ndim == 2, f"prediction must be 2D; got {prediction_m.shape}"
    return prediction_m, len(tile_cache)


def infer(
    model_fp: str | Path,
    depth_lr_fp: str | Path,
    dem_hr_fp: str | Path,
    output_fp: str | Path,
    max_depth: float | None = None,
    dem_pct_clip: float | None = None,
    window_method: str = "feather",
    tile_overlap: int | None = None,
    tile_size: int | None = None,
    logger=None,
) -> dict[str, object]:
    """
    Run one tiled raster inference pass.

    Parameters
    ----------
    model_fp:
        Path to the ONNX model file.
    depth_lr_fp:
        Path to low-resolution depth raster (single band).
    dem_hr_fp:
        Path to high-resolution DEM raster (single band).
    output_fp:
        Path to output high-resolution depth prediction raster.
    max_depth:
        Optional max depth override for log scaling.
    dem_pct_clip:
        Optional DEM percentile clip override when train stats are incomplete.
    window_method:
        Mosaicing strategy: hard or feather.
    tile_overlap:
        Overlap in LR pixels for feather mode.
    tile_size:
        Optional LR tile override; must match model contract.
    logger:
        Optional logger instance.

    Returns
    -------
    dict
        Metadata dictionary with output path and runtime diagnostics.
    """
    start = time.perf_counter()
    log = logger or logging.getLogger(__name__)
    model_path = Path(model_fp).expanduser().resolve()
    depth_lr_path = Path(depth_lr_fp).expanduser().resolve()
    dem_hr_path = Path(dem_hr_fp).expanduser().resolve()
    out_path = Path(output_fp).expanduser().resolve()
    assert model_path.exists(), f"model file does not exist: {model_path}"
    assert depth_lr_path.exists(), f"low-res depth raster does not exist: {depth_lr_path}"
    assert dem_hr_path.exists(), f"DEM raster does not exist: {dem_hr_path}"
    window_method = (window_method or "feather").strip().lower()
    assert window_method in {"hard", "feather"}, f"unsupported window_method={window_method}"

    log.info(
        f"starting raster inference with model\n    {model_path}\n"
        f"depth_lr\n    {depth_lr_path}\n"
        f"dem_hr\n    {dem_hr_path}\n"
        f"output\n    {out_path}"
    )
    depth_lr_raw, _, depth_lr_raw_profile = _read_single_band_raster(depth_lr_path)
    dem_hr_raw, _, dem_hr_raw_profile = _read_single_band_raster(dem_hr_path)
    log.info(
        "raw inputs\n"
        f"  depth_lr shape={depth_lr_raw.shape} res={_pixel_size_m(depth_lr_raw_profile)} m/pix\n"
        f"  dem_hr  shape={dem_hr_raw.shape} res={_pixel_size_m(dem_hr_raw_profile)} m/pix"
    )

    preprocess_cfg = resolve_preprocess_config(
        model_path,
        max_depth=max_depth,
        dem_pct_clip=dem_pct_clip,
        logger=log,
    )

    from floodsr.engine.ort import EngineORT

    engine = EngineORT(model_path, logger=log)
    assert engine.contract is not None, "engine contract must be available"

    contract_scale = int(engine.contract.scale)
    contract_lr_tile = int(engine.contract.depth_lr_hwc[0])
    contract_hr_tile = int(engine.contract.dem_hr_hwc[0])

    model_scale = (
        int(preprocess_cfg["scale"]) if isinstance(preprocess_cfg.get("scale"), int | float) else contract_scale
    )
    if model_scale != contract_scale:
        log.warning(f"using contract scale {contract_scale} over configured scale {model_scale}")
        model_scale = contract_scale

    model_lr_tile = (
        int(preprocess_cfg["lr_tile"])
        if isinstance(preprocess_cfg.get("lr_tile"), int | float)
        else contract_lr_tile
    )
    if model_lr_tile != contract_lr_tile:
        log.warning(
            f"model config LR tile {model_lr_tile} overrides contract tile {contract_lr_tile}; "
            "using contract tile for strict model shape checks."
        )
        model_lr_tile = contract_lr_tile

    if tile_size is not None:
        tile_size = int(tile_size)
        if tile_size != contract_lr_tile:
            raise AssertionError(
                f"tile_size override {tile_size} does not match model LR tile {contract_lr_tile}"
            )
        model_lr_tile = tile_size

    if model_lr_tile * model_scale != contract_hr_tile:
        raise AssertionError(
            f"model tile mismatch: LR tile {model_lr_tile} x scale {model_scale} "
            f"!= contract HR tile {contract_hr_tile}"
        )

    overlap_lr = int(tile_overlap) if tile_overlap is not None else contract_lr_tile // 4
    if overlap_lr < 0:
        raise AssertionError(f"tile_overlap must be >= 0; got {overlap_lr}")

    with tempfile.TemporaryDirectory(prefix="floodsr-prep-") as prepped_dir:
        prepped = write_prepared_rasters(
            depth_lr_fp=depth_lr_path,
            dem_hr_fp=dem_hr_path,
            scale=model_scale,
            out_dir=prepped_dir,
            logger=log,
        )
        log.info(
            "preprocessing complete\n"
            f"  scale={model_scale} (HR/LR ratio)\n"
            f"  aligned depth shape={prepped['depth_lr_shape']} "
            f"resampled={prepped['resampled']}\n"
            f"  aligned dem shape={prepped['dem_hr_shape']}\n"
            f"  max_depth={float(preprocess_cfg['max_depth'])} "
            f"dem_pct_clip={float(preprocess_cfg['dem_pct_clip'])}"
        )
        prediction_m, tile_cache_size = infer_from_prepared_inputs(
            engine=engine,
            depth_lr_fp=prepped["depth_lr_prepared_fp"],
            dem_hr_fp=prepped["dem_hr_prepared_fp"],
            preprocess_cfg=preprocess_cfg,
            model_lr_tile=model_lr_tile,
            model_scale=model_scale,
            contract_hr_tile=contract_hr_tile,
            window_method=window_method,
            overlap_lr=overlap_lr,
            logger=log,
        )

        _, _, dem_prepared_profile = _read_single_band_raster(prepped["dem_hr_prepared_fp"])
        output_profile = dem_prepared_profile.copy()
        output_profile.update(dtype="float32", count=1)
        output_profile.pop("blockxsize", None)
        output_profile.pop("blockysize", None)

        out_written_fp = _write_single_band_raster(out_path, prediction_m, output_profile)

    runtime_s = time.perf_counter() - start
    log.info(f"finished raster inference in {runtime_s:.3f}s; wrote output to\n    {out_written_fp}")
    return {
        "output_fp": str(out_written_fp),
        "runtime_s": float(runtime_s),
        "preprocess": {
            "max_depth": float(preprocess_cfg["max_depth"]),
            "dem_pct_clip": float(preprocess_cfg["dem_pct_clip"]),
            "dem_ref_stats": preprocess_cfg["dem_ref_stats"],
            "dem_stats_used": {
                "p_clip": float(
                    (preprocess_cfg.get("dem_ref_stats") or {}).get(
                        "p_clip", float(preprocess_cfg["dem_pct_clip"])
                    )
                ),
            },
            "window_method": window_method,
            "tile_overlap_lr": overlap_lr,
            "tile_size_lr": model_lr_tile,
            "tile_size_hr": contract_hr_tile,
            "model_scale": model_scale,
            "tile_cache_size": tile_cache_size,
            "input_shape": {
                "crop_height": int(prediction_m.shape[0]),
                "crop_width": int(prediction_m.shape[1]),
                "aligned_depth_shape": [int(x) for x in prepped["depth_lr_shape"]],
                "aligned_dem_shape": [int(x) for x in prepped["dem_hr_shape"]],
            },
            "prepared_inputs": {
                "depth_lr_prepared_fp": str(prepped["depth_lr_prepared_fp"]),
                "dem_hr_prepared_fp": str(prepped["dem_hr_prepared_fp"]),
                "prepped_depth_was_resampled": bool(prepped["resampled"]),
            },
        },
    }
