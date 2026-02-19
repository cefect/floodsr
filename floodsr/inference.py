"""Inference utilities aligned with the notebook proof-of-concept contract."""

import json, logging, time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def _as_numeric_np_array(
    arr: np.ndarray,
    name: str,
    min_rank: int = 1,
    allow_ranks: Optional[Tuple[int, ...]] = None,
    require_single_channel_last_dim: bool = False,
) -> np.ndarray:
    """Convert input to NumPy array and validate dtype/rank constraints."""
    out = np.asarray(arr)
    if out.dtype == np.bool_ or not np.issubdtype(out.dtype, np.number):
        raise AssertionError(f"{name} must have numeric dtype; got {out.dtype}")

    rank = int(out.ndim)
    if allow_ranks is not None:
        if rank not in allow_ranks:
            raise AssertionError(f"{name} rank must be one of {allow_ranks}; got rank {rank} shape {out.shape}")
    elif rank < min_rank:
        raise AssertionError(f"{name} rank must be >= {min_rank}; got rank {rank} shape {out.shape}")

    if require_single_channel_last_dim and rank >= 3 and out.shape[-1] != 1:
        raise AssertionError(f"{name} last dim must be 1 for rank >=3; got shape {out.shape}")

    if not np.all(np.isfinite(out)):
        raise AssertionError(f"{name} must contain only finite values")
    return out


def _parse_dem_normalization_stats(ref_stats: Dict[str, float]) -> Tuple[float, float, float]:
    """Validate and unpack DEM normalization stats dictionary."""
    required = {"p_clip", "dem_min", "dem_max"}
    missing = required.difference(ref_stats.keys())
    if missing:
        raise AssertionError(f"DEM ref_stats missing keys: {sorted(missing)}")

    p_clip = float(ref_stats["p_clip"])
    dem_min = float(ref_stats["dem_min"])
    dem_max = float(ref_stats["dem_max"])

    if not (np.isfinite(p_clip) and np.isfinite(dem_min) and np.isfinite(dem_max)):
        raise AssertionError("DEM ref_stats values must be finite")
    if p_clip < 0:
        raise AssertionError(f"DEM p_clip must be >= 0; got {p_clip}")
    if dem_min > dem_max:
        raise AssertionError(f"DEM dem_min must be <= dem_max; got min={dem_min} max={dem_max}")
    if (dem_max - dem_min) <= 0:
        raise AssertionError(f"DEM range must be > 0; got min={dem_min}, max={dem_max}")
    return p_clip, dem_min, dem_max


def normalize_dem_with_stats_np(
    arr: np.ndarray,
    p_clip: float,
    dem_min: float,
    dem_max: float,
) -> np.ndarray:
    """Normalize DEM with already-computed clip/min/max stats."""
    if not (np.isfinite(p_clip) and np.isfinite(dem_min) and np.isfinite(dem_max)):
        raise AssertionError("p_clip, dem_min, and dem_max must be finite")

    dem_range = dem_max - dem_min
    if dem_range <= 0:
        raise AssertionError(f"DEM range must be > 0; got min={dem_min}, max={dem_max}")

    arr_np = _as_numeric_np_array(
        arr,
        "dem_arr",
        allow_ranks=(2, 3, 4),
        require_single_channel_last_dim=True,
    ).astype(np.float32, copy=False)

    arr_clipped = np.clip(arr_np, 0.0, float(p_clip))
    arr_norm = (arr_clipped - float(dem_min)) / float(dem_range)
    arr_norm = np.clip(arr_norm, 0.0, 1.0)
    return arr_norm.astype(np.float32, copy=False)


def normalize_dem(
    arr: Optional[np.ndarray],
    pct_clip: float = 95.0,
    ref_stats: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
    """Clip DEM and min-max normalize to [0,1] with notebook-compatible rules."""
    if arr is None:
        return None, None

    if ref_stats is None:
        pct_clip = float(pct_clip)
        if not np.isfinite(pct_clip) or not (0 < pct_clip <= 100):
            raise AssertionError(f"dem_pct_clip must be finite and in (0, 100]; got {pct_clip}")

        arr_np = _as_numeric_np_array(
            arr,
            "dem_arr",
            allow_ranks=(2, 3, 4),
            require_single_channel_last_dim=True,
        ).astype(np.float32, copy=False)
        arr_np = np.clip(arr_np, 0.0, None)
        p_clip = float(np.nanpercentile(arr_np, pct_clip))
        arr_for_stats = np.clip(arr_np, 0.0, p_clip)
        dem_min = float(np.nanmin(arr_for_stats))
        dem_max = float(np.nanmax(arr_for_stats))
    else:
        p_clip, dem_min, dem_max = _parse_dem_normalization_stats(ref_stats)

    arr_norm = normalize_dem_with_stats_np(arr, p_clip=p_clip, dem_min=dem_min, dem_max=dem_max)
    return arr_norm, {"p_clip": p_clip, "dem_min": dem_min, "dem_max": dem_max}


def _depth_log1p_denom(max_depth: float) -> float:
    """Compute and validate denominator used by depth log1p scaling."""
    max_depth = float(max_depth)
    if not np.isfinite(max_depth) or max_depth <= 0:
        raise AssertionError(f"max_depth must be finite and > 0; got {max_depth}")

    denom = float(np.log1p(max_depth))
    if not np.isfinite(denom) or denom <= 0:
        raise AssertionError(f"log1p(max_depth) must be finite and > 0; got {denom}")
    return denom


def scale_depth_log1p_np(arr: Optional[np.ndarray], max_depth: float) -> Optional[np.ndarray]:
    """Clip depth values and apply log1p normalization."""
    if arr is None:
        return None

    denom = _depth_log1p_denom(max_depth)
    arr_np = _as_numeric_np_array(arr, "depth_arr", min_rank=1).astype(np.float32, copy=False)
    arr_np = np.clip(arr_np, 0.0, float(max_depth))
    scaled = np.log1p(arr_np) / denom
    scaled = np.clip(scaled, 0.0, 1.0)
    return scaled.astype(np.float32, copy=False)


def invert_depth_log1p_np(arr: Optional[np.ndarray], max_depth: float) -> Optional[np.ndarray]:
    """Invert normalized log1p depth values back to depth units."""
    if arr is None:
        return None

    denom = _depth_log1p_denom(max_depth)
    arr_np = _as_numeric_np_array(arr, "normalized_depth_arr", min_rank=1).astype(np.float32, copy=False)
    arr_np = np.clip(arr_np, 0.0, 1.0)
    inv = np.expm1(arr_np * denom)
    inv = np.clip(inv, 0.0, float(max_depth))
    return inv.astype(np.float32, copy=False)


def replace_nodata_with_zero(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    """Replace nodata values with 0.0 for notebook-compatible preprocessing."""
    arr_np = np.asarray(arr, dtype=np.float32)
    if nodata is None:
        return arr_np
    return np.where(np.isclose(arr_np, nodata), 0.0, arr_np).astype(np.float32, copy=False)


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


def load_train_config(model_fp: str | Path, logger=None) -> dict | None:
    """Load train_config.json from model directory when present."""
    log = logger or logging.getLogger(__name__)
    model_path = Path(model_fp).expanduser().resolve()
    train_config_fp = model_path.parent / "train_config.json"
    if not train_config_fp.exists():
        log.debug(f"train config not found for model\n    {model_path}")
        return None
    log.debug(f"loaded train config from\n    {train_config_fp}")
    return json.loads(train_config_fp.read_text(encoding="utf-8"))


def resolve_preprocess_config(
    model_fp: str | Path,
    max_depth: float | None = None,
    dem_pct_clip: float | None = None,
    logger=None,
) -> dict[str, object]:
    """Resolve preprocessing constants from model-adjacent train_config when available."""
    log = logger or logging.getLogger(__name__)
    model_path = Path(model_fp).expanduser().resolve()
    assert model_path.exists(), f"model file does not exist: {model_path}"

    resolved_max_depth = 5.0 if max_depth is None else float(max_depth)
    resolved_dem_pct_clip = 95.0 if dem_pct_clip is None else float(dem_pct_clip)
    dem_ref_stats = None
    train_cfg = load_train_config(model_path, logger=log)
    if train_cfg is not None:
        if max_depth is None and train_cfg.get("max_depth") is not None:
            resolved_max_depth = float(train_cfg["max_depth"])
        if dem_pct_clip is None and train_cfg.get("dem_pct_clip") is not None:
            resolved_dem_pct_clip = float(train_cfg["dem_pct_clip"])
        dem_stats_cfg = train_cfg.get("dem_stats") or {}
        required_keys = {"p_clip", "dem_min", "dem_max"}
        if required_keys.issubset(dem_stats_cfg):
            dem_ref_stats = {k: float(dem_stats_cfg[k]) for k in sorted(required_keys)}

    log.debug(
        f"resolved preprocessing config: max_depth={resolved_max_depth}, "
        f"dem_pct_clip={resolved_dem_pct_clip}, has_dem_ref_stats={dem_ref_stats is not None}"
    )
    return {
        "max_depth": resolved_max_depth,
        "dem_pct_clip": resolved_dem_pct_clip,
        "dem_ref_stats": dem_ref_stats,
    }


def _read_single_band_raster(fp: str | Path) -> tuple[np.ndarray, float | None, dict]:
    """Read a single-band raster and return array, nodata, and profile."""
    import rasterio

    path = Path(fp).expanduser().resolve()
    assert path.exists(), f"raster does not exist: {path}"
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        nodata = ds.nodata
        profile = ds.profile.copy()
    return arr, nodata, profile


def _write_single_band_raster(fp: str | Path, arr: np.ndarray, profile: dict) -> Path:
    """Write a float32 single-band raster with a copied profile."""
    import rasterio
    from rasterio.drivers import driver_from_extension

    path = Path(fp).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved_driver = driver_from_extension(path)
    except ValueError:
        resolved_driver = "GTiff"
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, driver=resolved_driver)
    if not bool(out_profile.get("tiled", False)):
        out_profile.pop("blockxsize", None)
        out_profile.pop("blockysize", None)
    with rasterio.open(path, "w", **out_profile) as ds:
        ds.write(arr.astype(np.float32, copy=False), 1)
    return path


def infer_geotiff(
    model_fp: str | Path,
    depth_lr_fp: str | Path,
    dem_hr_fp: str | Path,
    output_fp: str | Path,
    max_depth: float | None = None,
    dem_pct_clip: float | None = None,
    logger=None,
) -> dict[str, object]:
    """
    Run one GeoTIFF inference using the notebook-aligned model I/O pipeline.

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
    logger:
        Optional logger instance.

    Returns
    -------
    dict
        Metadata dictionary with output path, preprocessing config, and runtime diagnostics.
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

    log.info(
        f"starting GeoTIFF inference with model\n    {model_path}\n"
        f"depth_lr\n    {depth_lr_path}\n"
        f"dem_hr\n    {dem_hr_path}\n"
        f"output\n    {out_path}"
    )
    preprocess_cfg = resolve_preprocess_config(
        model_path,
        max_depth=max_depth,
        dem_pct_clip=dem_pct_clip,
        logger=log,
    )
    depth_lr_raw, depth_lr_nodata, _depth_lr_profile = _read_single_band_raster(depth_lr_path)
    dem_hr_raw, dem_hr_nodata, dem_hr_profile = _read_single_band_raster(dem_hr_path)

    # Run model inference through the ORT engine with PoC-compatible preprocessing.
    from floodsr.engine.ort import EngineORT

    engine = EngineORT(model_path, logger=log)
    run_result = engine.run_tile(
        depth_lr_raw,
        dem_hr_raw,
        max_depth=float(preprocess_cfg["max_depth"]),
        dem_pct_clip=float(preprocess_cfg["dem_pct_clip"]),
        dem_ref_stats=preprocess_cfg["dem_ref_stats"],
        depth_lr_nodata=depth_lr_nodata,
        dem_hr_nodata=dem_hr_nodata,
    )
    prediction_m = run_result["prediction_m"]
    assert prediction_m.ndim == 2, f"prediction must be 2D; got {prediction_m.shape}"
    out_written_fp = _write_single_band_raster(out_path, prediction_m, dem_hr_profile)

    runtime_s = time.perf_counter() - start
    log.info(f"finished GeoTIFF inference in {runtime_s:.3f}s; wrote output to\n    {out_written_fp}")
    return {
        "output_fp": str(out_written_fp),
        "runtime_s": float(runtime_s),
        "preprocess": {
            "max_depth": float(preprocess_cfg["max_depth"]),
            "dem_pct_clip": float(preprocess_cfg["dem_pct_clip"]),
            "dem_ref_stats": preprocess_cfg["dem_ref_stats"],
            "dem_stats_used": run_result["dem_stats_used"],
        },
    }
