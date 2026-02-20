"""Inference utilities aligned with the notebook proof-of-concept contract."""

import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
        if np.isclose(dem_range, 0.0) and np.isclose(dem_min, 0.0):
            # All-zero DEM tiles can occur at padded/nodata edges after clipping; keep a
            # stable normalized representation instead of failing the full inference pass.
            return np.zeros_like(_as_numeric_np_array(
                arr,
                "dem_arr",
                allow_ranks=(2, 3, 4),
                require_single_channel_last_dim=True,
            ).astype(np.float32, copy=False))

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
    resolved_lr_tile = None
    resolved_scale = None
    resolved_dem_resolution = None
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
        input_shape = train_cfg.get("input_shape")
        if isinstance(input_shape, (tuple, list)) and len(input_shape) >= 2:
            lr_h = input_shape[0]
            if isinstance(lr_h, (int, float)) and float(lr_h).is_integer():
                resolved_lr_tile = int(lr_h)
        if train_cfg.get("upscale") is not None:
            resolved_scale = int(train_cfg["upscale"])
        if train_cfg.get("dem_fp"):
            dem_fp = str(train_cfg.get("dem_fp"))
            match = re.search(r"(?:^|[_/])([0-9]{2,})_?dem", dem_fp)
            if match is not None:
                resolved_dem_resolution = int(match.group(1))

    log.debug(
        f"resolved preprocessing config: max_depth={resolved_max_depth}, "
        f"dem_pct_clip={resolved_dem_pct_clip}, has_dem_ref_stats={dem_ref_stats is not None}, "
        f"lr_tile={resolved_lr_tile}, scale={resolved_scale}, "
        f"model_dem_resolution={resolved_dem_resolution}"
    )
    return {
        "max_depth": resolved_max_depth,
        "dem_pct_clip": resolved_dem_pct_clip,
        "dem_ref_stats": dem_ref_stats,
        "lr_tile": resolved_lr_tile,
        "scale": resolved_scale,
        "model_dem_resolution": resolved_dem_resolution,
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


def _build_tile_starts(total_size: int, tile_size: int, stride: int) -> list[int]:
    """Build tile origin positions that always include trailing-edge coverage."""
    assert total_size > 0, f"total_size must be > 0; got {total_size}"
    assert tile_size > 0, f"tile_size must be > 0; got {tile_size}"
    assert stride > 0, f"stride must be > 0; got {stride}"
    starts = list(range(0, max(total_size - tile_size + 1, 1), stride))
    last_start = total_size - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _align_depth_and_dem_inputs(
    depth_lr_fp: str | Path,
    dem_hr_fp: str | Path,
    scale: int,
    logger=None,
) -> dict[str, Any]:
    """Read inputs, enforce geospatial compatibility, and align them for model tiling."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import from_bounds
    from rasterio.transform import from_bounds as bounds_to_transform

    log = logger or logging.getLogger(__name__)
    assert scale > 0, f"scale must be > 0; got {scale}"
    depth_path = Path(depth_lr_fp).expanduser().resolve()
    dem_path = Path(dem_hr_fp).expanduser().resolve()
    assert depth_path.exists(), f"low-res depth raster does not exist: {depth_path}"
    assert dem_path.exists(), f"hires DEM raster does not exist: {dem_path}"

    with rasterio.open(depth_path) as depth_ds, rasterio.open(dem_path) as dem_ds:
        assert depth_ds.count == 1, f"depth raster must have 1 band; got {depth_ds.count}"
        assert dem_ds.count == 1, f"DEM raster must have 1 band; got {dem_ds.count}"

        depth_crs = depth_ds.crs
        dem_crs = dem_ds.crs
        if depth_crs is None:
            assert dem_crs is not None, "both rasters must include CRS when depth CRS is missing"
            depth_crs = dem_crs
            log.warning(
                "assigning missing depth CRS from DEM CRS\n"
                f"    depth={depth_path}\n"
                f"    dem={dem_path}"
            )
        assert dem_crs is not None, "both rasters must define CRS"
        assert depth_crs == dem_crs, (
            f"CRS mismatch\n"
            f"    depth={depth_crs}\n"
            f"    dem={dem_crs}"
        )
        assert depth_crs.is_projected, f"CRS must be projected; got {depth_crs}"

        depth_res = (abs(float(depth_ds.res[0])), abs(float(depth_ds.res[1])))
        dem_res = (abs(float(dem_ds.res[0])), abs(float(dem_ds.res[1])))
        if not np.isclose(depth_res[0], depth_res[1]):
            log.warning(f"depth pixels are not square: res={depth_res}")
        if not np.isclose(dem_res[0], dem_res[1]):
            log.warning(f"DEM pixels are not square: res={dem_res}")

        lr_bounds = depth_ds.bounds
        dem_bounds = dem_ds.bounds
        if not all(np.isclose(lr_bounds, dem_bounds, atol=1e-6, rtol=0.0)):
            log.warning(
                "input bounds differ; clipping DEM to depth raster bounds.\n"
                f"    depth={lr_bounds}\n"
                f"    dem={dem_bounds}"
            )

        dem_window = from_bounds(*lr_bounds, dem_ds.transform).round_offsets().round_lengths()
        dem_crop = dem_ds.read(1, window=dem_window).astype(np.float32, copy=False)
        assert dem_crop.size > 0, f"clipped DEM is empty for bounds {lr_bounds}"
        dem_crop_transform = dem_ds.window_transform(dem_window)
        dem_nodata = dem_ds.nodata
        depth_nodata = depth_ds.nodata
        depth_lr = depth_ds.read(1).astype(np.float32, copy=False)
        depth_bounds = lr_bounds
        depth_transform = depth_ds.transform
        depth_crs = depth_ds.crs
        depth_profile = depth_ds.profile.copy()
        dem_profile = dem_ds.profile.copy()
        dem_bounds = dem_ds.bounds

    if not np.isfinite(dem_crop).all():
        raise AssertionError("DEM contains non-finite values after clipping")
    if not np.isfinite(depth_lr).all():
        raise AssertionError("low-res depth contains non-finite values")
    if depth_lr.min() < 0.0:
        raise AssertionError(f"low-res depth has negative values: min={float(depth_lr.min())}")

    dem_h, dem_w = dem_crop.shape
    crop_h = dem_h - (dem_h % scale)
    crop_w = dem_w - (dem_w % scale)
    assert crop_h > 0 and crop_w > 0, f"cropped DEM has invalid size {(crop_h, crop_w)}"
    if (crop_h, crop_w) != (dem_h, dem_w):
        log.debug(f"cropping DEM from {(dem_h, dem_w)} to {(crop_h, crop_w)} for scale alignment")
    dem_crop = dem_crop[:crop_h, :crop_w]

    target_lr_h = crop_h // scale
    target_lr_w = crop_w // scale
    assert target_lr_h > 0 and target_lr_w > 0, f"target LR shape invalid {(target_lr_h, target_lr_w)}"
    was_resampled = (target_lr_h != depth_lr.shape[0] or target_lr_w != depth_lr.shape[1])
    if was_resampled:
        log.debug(
            f"resampling low-res depth from {depth_lr.shape} to {(target_lr_h, target_lr_w)} "
            f"to enforce model-scale={scale}"
        )
        depth_lr_aligned = np.empty((target_lr_h, target_lr_w), dtype=np.float32)
        depth_lr_transform = bounds_to_transform(
            *depth_bounds,
            width=target_lr_w,
            height=target_lr_h,
        )
        reproject(
            source=depth_lr,
            destination=depth_lr_aligned,
            src_transform=depth_transform,
            src_crs=depth_crs,
            src_nodata=depth_nodata,
            dst_transform=depth_lr_transform,
            dst_crs=depth_crs,
            dst_nodata=depth_nodata,
            resampling=Resampling.bilinear,
            num_threads=1,
        )
        depth_lr = depth_lr_aligned
    return {
        "depth_lr": depth_lr,
        "depth_lr_nodata": depth_nodata,
        "dem_hr": dem_crop,
        "dem_hr_nodata": dem_nodata,
        "depth_lr_profile": depth_profile,
        "dem_profile": dem_profile,
        "dem_crop_transform": dem_crop_transform,
        "crop_shape": (crop_h, crop_w),
        "resampled": was_resampled,
        "depth_bounds": depth_bounds,
    }
def infer_geotiff(
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
    Run one GeoTIFF inference using tiled model execution.

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
    window_method = (window_method or "feather").strip().lower()
    assert window_method in {"hard", "feather"}, f"unsupported window_method={window_method}"

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

    aligned = _align_depth_and_dem_inputs(
        depth_lr_fp=depth_lr_path,
        dem_hr_fp=dem_hr_path,
        scale=model_scale,
        logger=log,
    )
    depth_lr_raw = aligned["depth_lr"]
    depth_lr_nodata = aligned["depth_lr_nodata"]
    dem_hr_raw = aligned["dem_hr"]
    dem_hr_nodata = aligned["dem_hr_nodata"]
    crop_h, crop_w = aligned["crop_shape"]

    output_profile = aligned["dem_profile"].copy()
    output_profile.update(
        {
            "height": int(crop_h),
            "width": int(crop_w),
            "transform": aligned["dem_crop_transform"],
        }
    )
    output_profile.update(dtype="float32", count=1)
    output_profile.pop("blockxsize", None)
    output_profile.pop("blockysize", None)

    if depth_lr_raw.min() > float(preprocess_cfg["max_depth"]):
        log.warning("low-res depth values exceed max_depth; model preprocessing will clip them.")

    pad_h = (int(math.ceil(crop_h / contract_hr_tile)) * contract_hr_tile) - crop_h
    pad_w = (int(math.ceil(crop_w / contract_hr_tile)) * contract_hr_tile) - crop_w

    depth_pad_val = 0.0 if depth_lr_nodata is None else float(depth_lr_nodata)
    dem_pad_val = 0.0 if dem_hr_nodata is None else float(dem_hr_nodata)
    dem_pad = np.pad(
        dem_hr_raw,
        ((0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=dem_pad_val,
    )
    depth_pad = np.pad(
        depth_lr_raw,
        ((0, pad_h // model_scale), (0, pad_w // model_scale)),
        mode="constant",
        constant_values=depth_pad_val,
    )

    hr_pad_h, hr_pad_w = dem_pad.shape
    depth_pad_h, depth_pad_w = depth_pad.shape
    assert depth_pad_h == hr_pad_h // model_scale and depth_pad_w == hr_pad_w // model_scale, (
        f"depth pad shape {(depth_pad_h, depth_pad_w)} incompatible with HR pad {(hr_pad_h, hr_pad_w)}"
    )

    tile_cache: dict[tuple[int, int], np.ndarray] = {}

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
            max_depth=float(preprocess_cfg["max_depth"]),
            dem_pct_clip=float(preprocess_cfg["dem_pct_clip"]),
            dem_ref_stats=preprocess_cfg["dem_ref_stats"],
            depth_lr_nodata=depth_lr_nodata,
            dem_hr_nodata=dem_hr_nodata,
        )
        pred = run_result["prediction_m"]
        assert pred.shape == (contract_hr_tile, contract_hr_tile), (
            f"prediction shape {pred.shape} != {(contract_hr_tile, contract_hr_tile)}"
        )
        tile_cache[key] = pred
        return pred

    if window_method == "hard":
        sr_pad = np.zeros_like(dem_pad, dtype=np.float32)
        nonoverlap_y = list(range(0, hr_pad_h, contract_hr_tile))
        nonoverlap_x = list(range(0, hr_pad_w, contract_hr_tile))
        log.debug(
            f"hard mosaicing over {len(nonoverlap_y) * len(nonoverlap_x)} tiles size={contract_hr_tile}"
        )
        for y0 in nonoverlap_y:
            for x0 in nonoverlap_x:
                pred_np = _predict_tile(y0, x0)
                sr_pad[y0 : y0 + contract_hr_tile, x0 : x0 + contract_hr_tile] = pred_np

    elif window_method == "feather":
        overlap_hr = overlap_lr * model_scale
        stride_hr = contract_hr_tile - overlap_hr
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
        log.debug(
            f"feather mosaicing with {len(y_starts)}x{len(x_starts)} tiles and overlap={overlap_hr} px"
        )
        for yi, y0 in enumerate(y_starts):
            for xi, x0 in enumerate(x_starts):
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

    out_written_fp = _write_single_band_raster(out_path, prediction_m, output_profile)
    runtime_s = time.perf_counter() - start
    log.info(f"finished GeoTIFF inference in {runtime_s:.3f}s; wrote output to\n    {out_written_fp}")
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
            "tile_cache_size": len(tile_cache),
            "input_shape": {
                "crop_height": int(crop_h),
                "crop_width": int(crop_w),
                "aligned_depth_shape": [int(x) for x in depth_lr_raw.shape],
                "aligned_dem_shape": [int(x) for x in dem_hr_raw.shape],
            },
        },
    }
