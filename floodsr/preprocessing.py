"""Preprocessing utilities for FloodSR raster inference."""

import json
import logging
import re
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
    """Convert input to NumPy and validate numeric dtype and rank constraints."""
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
    """Validate and unpack DEM normalization statistics."""
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
    """Normalize DEM values using explicit min/max/clip stats."""
    if not (np.isfinite(p_clip) and np.isfinite(dem_min) and np.isfinite(dem_max)):
        raise AssertionError("p_clip, dem_min, and dem_max must be finite")

    dem_range = dem_max - dem_min
    if dem_range <= 0:
        if np.isclose(dem_range, 0.0) and np.isclose(dem_min, 0.0):
            # Pinned DEMs can appear in padded/nodata edges; keep stable output here.
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
    """Clip and normalize DEM arrays to [0, 1]."""
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
    """Compute normalization denominator for depth log1p scaling."""
    max_depth = float(max_depth)
    if not np.isfinite(max_depth) or max_depth <= 0:
        raise AssertionError(f"max_depth must be finite and > 0; got {max_depth}")

    denom = float(np.log1p(max_depth))
    if not np.isfinite(denom) or denom <= 0:
        raise AssertionError(f"log1p(max_depth) must be finite and > 0; got {denom}")
    return denom


def scale_depth_log1p_np(arr: Optional[np.ndarray], max_depth: float) -> Optional[np.ndarray]:
    """Normalize depth values with log1p scaling."""
    if arr is None:
        return None

    denom = _depth_log1p_denom(max_depth)
    arr_np = _as_numeric_np_array(arr, "depth_arr", min_rank=1).astype(np.float32, copy=False)
    arr_np = np.clip(arr_np, 0.0, float(max_depth))
    scaled = np.log1p(arr_np) / denom
    scaled = np.clip(scaled, 0.0, 1.0)
    return scaled.astype(np.float32, copy=False)


def invert_depth_log1p_np(arr: Optional[np.ndarray], max_depth: float) -> Optional[np.ndarray]:
    """Invert log1p-normalized depth arrays to depth units."""
    if arr is None:
        return None

    denom = _depth_log1p_denom(max_depth)
    arr_np = _as_numeric_np_array(arr, "normalized_depth_arr", min_rank=1).astype(np.float32, copy=False)
    arr_np = np.clip(arr_np, 0.0, 1.0)
    inv = np.expm1(arr_np * denom)
    inv = np.clip(inv, 0.0, float(max_depth))
    return inv.astype(np.float32, copy=False)


def replace_nodata_with_zero(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    """Replace nodata values with zero for deterministic model inputs."""
    arr_np = np.asarray(arr, dtype=np.float32)
    if nodata is None:
        return arr_np
    return np.where(np.isclose(arr_np, nodata), 0.0, arr_np).astype(np.float32, copy=False)


def load_train_config(model_fp: str | Path, logger=None) -> dict | None:
    """Load `train_config.json` from the model directory if available."""
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
    """Resolve preprocessing defaults from a model's training config."""
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
    """Read a single-band raster from disk."""
    import rasterio

    path = Path(fp).expanduser().resolve()
    assert path.exists(), f"raster does not exist: {path}"
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        nodata = ds.nodata
        profile = ds.profile.copy()
    return arr, nodata, profile


def _write_single_band_raster(fp: str | Path, arr: np.ndarray, profile: dict, driver: str | None = None) -> Path:
    """Write a float32 single-band raster and return the output path."""
    import rasterio
    from rasterio.drivers import driver_from_extension

    path = Path(fp).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1)
    if driver is None:
        try:
            inferred_driver = driver_from_extension(path)
        except ValueError:
            inferred_driver = "GTiff"
        out_profile["driver"] = inferred_driver
    else:
        out_profile["driver"] = driver
    if not bool(out_profile.get("tiled", False)):
        out_profile.pop("blockxsize", None)
        out_profile.pop("blockysize", None)
    with rasterio.open(path, "w", **out_profile) as ds:
        ds.write(arr.astype(np.float32, copy=False), 1)
    return path


def _build_tile_starts(total_size: int, tile_size: int, stride: int) -> list[int]:
    """Build overlap-aware tile starts for raster coverage."""
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
    """Align raster inputs for model scale, clipping and resampling where needed."""
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
        depth_profile = depth_ds.profile.copy()
        dem_profile = dem_ds.profile.copy()

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
    depth_lr_transform = depth_transform
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
        "depth_lr_transform": depth_lr_transform,
        "depth_lr_profile": depth_profile,
        "dem_hr": dem_crop,
        "dem_hr_nodata": dem_nodata,
        "dem_crop_transform": dem_crop_transform,
        "dem_profile": dem_profile,
        "crop_shape": (crop_h, crop_w),
        "resampled": was_resampled,
    }


def write_prepared_rasters(
    depth_lr_fp: str | Path,
    dem_hr_fp: str | Path,
    *,
    scale: int,
    out_dir: str | Path,
    logger=None,
    depth_lr_prepared_fp: str | Path | None = None,
    dem_hr_prepared_fp: str | Path | None = None,
) -> dict[str, object]:
    """Write aligned/resized depth and DEM rasters to disk for inference."""
    log = logger or logging.getLogger(__name__)
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    aligned = _align_depth_and_dem_inputs(depth_lr_fp, dem_hr_fp, scale=scale, logger=log)

    depth_prepared_fp = Path(depth_lr_prepared_fp) if depth_lr_prepared_fp is not None else (
        out_dir / f"{Path(depth_lr_fp).stem}_prepped_depth.tif"
    )
    dem_prepared_fp = Path(dem_hr_prepared_fp) if dem_hr_prepared_fp is not None else (
        out_dir / f"{Path(dem_hr_fp).stem}_prepped_dem.tif"
    )
    depth_profile = aligned["depth_lr_profile"].copy()
    depth_profile.update(
        {
            "height": int(aligned["depth_lr"].shape[0]),
            "width": int(aligned["depth_lr"].shape[1]),
            "transform": aligned["depth_lr_transform"],
        }
    )
    dem_profile = aligned["dem_profile"].copy()
    dem_profile.update(
        {
            "height": int(aligned["crop_shape"][0]),
            "width": int(aligned["crop_shape"][1]),
            "transform": aligned["dem_crop_transform"],
        }
    )

    depth_prepared_path = _write_single_band_raster(depth_prepared_fp, aligned["depth_lr"], depth_profile)
    dem_prepared_path = _write_single_band_raster(dem_prepared_fp, aligned["dem_hr"], dem_profile)
    return {
        "depth_lr_prepared_fp": depth_prepared_path,
        "dem_hr_prepared_fp": dem_prepared_path,
        "depth_lr_profile": depth_profile,
        "dem_profile": dem_profile,
        "depth_lr_nodata": aligned["depth_lr_nodata"],
        "dem_hr_nodata": aligned["dem_hr_nodata"],
        "crop_shape": aligned["crop_shape"],
        "resampled": aligned["resampled"],
        "depth_lr_shape": tuple(aligned["depth_lr"].shape),
        "dem_hr_shape": tuple(aligned["dem_hr"].shape),
    }
