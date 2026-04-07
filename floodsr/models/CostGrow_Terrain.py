"""Built-in CostGrow terrain-penalty worker."""

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import scipy.ndimage
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, reproject

from floodsr.engine.pcraster_check import _check_pcraster
from floodsr.models.base import Model
from floodsr.preprocessing import (
    _build_single_band_profile,
    _read_single_band_raster,
    _write_single_band_raster,
    valid_mask_from_array,
)
from floodsr.tiling import iter_block_windows


def _resample_array_to_profile(
    src_arr: np.ndarray,
    src_profile: dict,
    dst_profile: dict,
    resampling: Resampling,
    src_nodata: float | None,
    dst_nodata: float,
) -> np.ndarray:
    """Reproject one array onto a destination profile."""
    dst_arr = np.full((int(dst_profile["height"]), int(dst_profile["width"])), dst_nodata, dtype=np.float32)
    reproject(
        source=src_arr.astype(np.float32, copy=False),
        destination=dst_arr,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        resampling=resampling,
    )
    return dst_arr


def _fill_nearest_unmasked(masked: np.ma.MaskedArray, method: str = "distance_transform_cdt", **kwargs) -> np.ndarray:
    """Fill masked cells with nearest unmasked values."""
    assert isinstance(masked, np.ma.MaskedArray), type(masked)
    if not masked.mask.any():
        return masked.data.copy()
    assert not masked.mask.all(), "array is fully masked"
    transform = getattr(scipy.ndimage, method)
    indices = transform(masked.mask.astype(int), return_indices=True, return_distances=False, **kwargs)
    filled = masked.data.copy()
    filled[masked.mask] = masked.data[tuple(indices[:, masked.mask])]
    return filled


def _compute_cost_surface(wse_filled_fine: np.ndarray, dem_fine: np.ndarray, dem_valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build terrain-penalty costs from filled fine-grid WSE and DEM."""
    delta = wse_filled_fine - dem_fine
    cost = np.where(delta > 0.0, 1.0, 1.0 + np.abs(delta)).astype(np.float32, copy=False)
    cost[~dem_valid_mask] = np.nan
    return cost, delta


def _filter_isolated(source_mask: np.ndarray, anchor_mask: np.ndarray) -> np.ndarray:
    """Keep only connected wet regions that intersect anchor wet cells."""
    assert source_mask.shape == anchor_mask.shape
    labels, _ = scipy.ndimage.label(source_mask.astype(np.uint8), structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    connected_labels = np.unique(labels[anchor_mask])
    connected_labels = connected_labels[connected_labels > 0]
    return np.isin(labels, connected_labels)


def _distance_fill_cost_pcraster(
    pcraster_module,
    wse_partial: np.ndarray,
    cost: np.ndarray,
    profile: dict,
) -> np.ndarray:
    """Fill masked WSE cells with PCRaster spreadzone over a terrain cost surface."""
    wet_mask = np.isfinite(wse_partial)
    if not wet_mask.any():
        raise AssertionError("wse_partial has no wet/source cells")
    if wet_mask.all():
        return wse_partial.copy()

    rows, cols = wse_partial.shape
    left, _, _, top = array_bounds(int(rows), int(cols), profile["transform"])
    res_x = abs(float(profile["transform"].a))
    res_y = abs(float(profile["transform"].e))
    cell_size = float(np.mean([res_x, res_y]))
    pcraster_module.setclone(int(rows), int(cols), cell_size, float(left), float(top))

    source_ids = np.zeros(wse_partial.shape, dtype=np.int32)
    source_idx = np.flatnonzero(wet_mask.ravel())
    source_ids.ravel()[source_idx] = np.arange(1, source_idx.size + 1, dtype=np.int32)

    lookup = np.full(source_idx.size + 1, np.nan, dtype=np.float64)
    lookup[1:] = wse_partial.ravel()[source_idx].astype(np.float64, copy=False)

    cost_mv = -9999.0
    cost_arr = cost.astype(np.float32, copy=True)
    invalid_cost = ~np.isfinite(cost_arr) | (cost_arr < 0.0)
    cost_arr[invalid_cost] = cost_mv

    points_map = pcraster_module.numpy2pcr(pcraster_module.Nominal, source_ids, -9999)
    friction_map = pcraster_module.numpy2pcr(pcraster_module.Scalar, cost_arr, cost_mv)
    zone_map = pcraster_module.spreadzone(points_map, 0, friction_map)
    zone_arr = pcraster_module.pcr2numpy(zone_map, 0).astype(np.int64, copy=False)

    max_zone = int(np.max(zone_arr))
    if max_zone > source_idx.size:
        raise AssertionError(f"spreadzone returned source id {max_zone} > {source_idx.size}")

    filled = lookup[zone_arr]
    filled[invalid_cost] = np.nan
    return filled.astype(np.float32, copy=False)


def _open_temp_memmap(tmp_dir: str | Path, stem: str, shape: tuple[int, int], dtype: np.dtype) -> np.memmap:
    """Create a disk-backed array for large-raster intermediate state."""
    path = Path(tmp_dir) / f"{stem}.npy"
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _close_temp_memmap(arr: np.memmap | None) -> None:
    """Flush and close one temp memmap so Windows can remove the backing file."""
    if arr is None:
        return
    arr.flush()
    mmap_obj = getattr(arr, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


def _window_slices(window) -> tuple[slice, slice]:
    """Convert a rasterio window into row/column slices."""
    row_off = int(window.row_off)
    col_off = int(window.col_off)
    height = int(window.height)
    width = int(window.width)
    return (
        slice(row_off, row_off + height),
        slice(col_off, col_off + width),
    )


def _run_costgrow_core_windowed(
    pcraster_module,
    depth_lr: np.ndarray,
    depth_lr_profile: dict,
    dem_fine_fp: str | Path,
    dem_fine_nodata: float | None,
    fine_profile: dict,
    output_fp: str | Path,
    out_nodata: float,
    min_depth_threshold: float,
    dp_coarse_pixel_max: int | None,
    decay_frac: float,
    distance_fill_method: str,
    distance_fill_kwargs: dict[str, Any] | None,
    show_progress: bool,
) -> tuple[Path, dict[str, Any]]:
    """Run CostGrow with disk-backed fine-grid intermediates and blockwise IO."""
    if distance_fill_kwargs is None:
        distance_fill_kwargs = {}

    fine_shape = (int(fine_profile["height"]), int(fine_profile["width"]))
    output_path = Path(output_fp).expanduser().resolve()
    output_profile = _build_single_band_profile(
        output_path,
        fine_profile,
        fine_shape[0],
        fine_shape[1],
        fine_profile["transform"],
    )
    output_profile["nodata"] = float(out_nodata)

    dem_valid_mask = None
    wse_fine = None
    filled_fine_wse = None
    wse_partials = None
    cost = None
    anchor_mask = None
    grown_wse = None
    distance_pixels = None
    source_mask = None
    labels = None
    tmp_dir_obj = tempfile.TemporaryDirectory(prefix="floodsr-costgrow-windowed-")
    try:
        tmp_dir = tmp_dir_obj.name
        dem_valid_mask = _open_temp_memmap(tmp_dir, "dem_valid_mask", fine_shape, np.uint8)
        with rasterio.open(dem_fine_fp) as dem_ds:
            for _, window in iter_block_windows(dem_ds, show_progress=show_progress, desc="costgrow dem valid pass"):
                row_slice, col_slice = _window_slices(window)
                dem_block = dem_ds.read(1, window=window).astype(np.float32, copy=False)
                dem_valid_mask[row_slice, col_slice] = valid_mask_from_array(dem_block, dem_fine_nodata).astype(
                    np.uint8,
                    copy=False,
                )

        dem_lr_arr = np.full(depth_lr.shape, np.nan, dtype=np.float32)
        dem_lr_valid_arr = np.zeros(depth_lr.shape, dtype=np.uint8)
        with rasterio.open(dem_fine_fp) as dem_ds:
            reproject(
                source=rasterio.band(dem_ds, 1),
                destination=dem_lr_arr,
                src_transform=dem_ds.transform,
                src_crs=dem_ds.crs,
                dst_transform=depth_lr_profile["transform"],
                dst_crs=depth_lr_profile["crs"],
                src_nodata=dem_fine_nodata,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
                num_threads=1,
            )
        reproject(
            source=dem_valid_mask,
            destination=dem_lr_valid_arr,
            src_transform=fine_profile["transform"],
            src_crs=fine_profile["crs"],
            dst_transform=depth_lr_profile["transform"],
            dst_crs=depth_lr_profile["crs"],
            src_nodata=0,
            dst_nodata=0,
            resampling=Resampling.nearest,
            num_threads=1,
        )
        dem_lr_valid_mask = dem_lr_valid_arr > 0

        coarse_wet = dem_lr_valid_mask & (depth_lr > float(min_depth_threshold))
        if not coarse_wet.any():
            raise AssertionError("depth_lr contains no wet/source cells above the minimum depth threshold")
        coarse_wse = np.where(coarse_wet, dem_lr_arr + depth_lr, np.nan).astype(np.float32, copy=False)
        filled_coarse_wse = _fill_nearest_unmasked(
            np.ma.MaskedArray(coarse_wse, mask=~np.isfinite(coarse_wse)),
            method=distance_fill_method,
            **distance_fill_kwargs,
        ).astype(np.float32, copy=False)
        filled_coarse_wse = np.where(np.isfinite(coarse_wse), coarse_wse, filled_coarse_wse)

        wse_fine = _open_temp_memmap(tmp_dir, "wse_fine", fine_shape, np.float32)
        filled_fine_wse = _open_temp_memmap(tmp_dir, "filled_fine_wse", fine_shape, np.float32)
        wse_fine.fill(np.nan)
        filled_fine_wse.fill(np.nan)
        reproject(
            source=coarse_wse,
            destination=wse_fine,
            src_transform=depth_lr_profile["transform"],
            src_crs=depth_lr_profile["crs"],
            dst_transform=fine_profile["transform"],
            dst_crs=fine_profile["crs"],
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            num_threads=1,
        )
        reproject(
            source=filled_coarse_wse,
            destination=filled_fine_wse,
            src_transform=depth_lr_profile["transform"],
            src_crs=depth_lr_profile["crs"],
            dst_transform=fine_profile["transform"],
            dst_crs=fine_profile["crs"],
            src_nodata=None,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            num_threads=1,
        )

        wse_partials = _open_temp_memmap(tmp_dir, "wse_partials", fine_shape, np.float32)
        cost = _open_temp_memmap(tmp_dir, "cost_surface", fine_shape, np.float32)
        anchor_mask = _open_temp_memmap(tmp_dir, "anchor_mask", fine_shape, np.uint8)
        wet_anchor_count = 0
        with rasterio.open(dem_fine_fp) as dem_ds:
            for _, window in iter_block_windows(dem_ds, show_progress=show_progress, desc="costgrow cost pass"):
                row_slice, col_slice = _window_slices(window)
                dem_block = dem_ds.read(1, window=window).astype(np.float32, copy=False)
                valid_block = dem_valid_mask[row_slice, col_slice] > 0
                wse_block = np.array(wse_fine[row_slice, col_slice], dtype=np.float32, copy=True)
                filled_block = np.array(filled_fine_wse[row_slice, col_slice], dtype=np.float32, copy=True)
                wet_above_ground = np.isfinite(wse_block) & valid_block & (wse_block > dem_block)
                partial_block = np.where(wet_above_ground, wse_block, np.nan).astype(np.float32, copy=False)
                cost_block, _ = _compute_cost_surface(filled_block, dem_block, valid_block)
                wse_partials[row_slice, col_slice] = partial_block
                cost[row_slice, col_slice] = cost_block
                anchor_mask[row_slice, col_slice] = wet_above_ground.astype(np.uint8, copy=False)
                wet_anchor_count += int(wet_above_ground.sum())
        if wet_anchor_count <= 0:
            raise AssertionError("wet-above-ground filtering fully masked the input wet cells")
        del wse_fine
        wse_fine = None
        del filled_fine_wse
        filled_fine_wse = None

        grown_wse = _open_temp_memmap(tmp_dir, "grown_wse", fine_shape, np.float32)
        grown_wse[:, :] = _distance_fill_cost_pcraster(pcraster_module, wse_partials, cost, fine_profile)
        distance_pixels = _open_temp_memmap(tmp_dir, "distance_pixels", fine_shape, np.int32)
        scipy.ndimage.distance_transform_cdt(
            anchor_mask == 0,
            return_distances=True,
            return_indices=False,
            distances=distance_pixels,
            **distance_fill_kwargs,
        )

        pixel_size_m = float(np.mean([abs(float(fine_profile["transform"].a)), abs(float(fine_profile["transform"].e))]))
        downscale = int(round(fine_shape[0] / depth_lr.shape[0]))
        max_distance_pixels = None if dp_coarse_pixel_max is None else int(dp_coarse_pixel_max) * max(downscale, 1)
        source_mask = _open_temp_memmap(tmp_dir, "source_mask", fine_shape, np.uint8)
        with rasterio.open(dem_fine_fp) as dem_ds:
            for _, window in iter_block_windows(dem_ds, show_progress=show_progress, desc="costgrow growth mask pass"):
                row_slice, col_slice = _window_slices(window)
                dem_block = dem_ds.read(1, window=window).astype(np.float32, copy=False)
                valid_block = dem_valid_mask[row_slice, col_slice] > 0
                anchor_block = anchor_mask[row_slice, col_slice] > 0
                grown_block = np.array(grown_wse[row_slice, col_slice], dtype=np.float32, copy=True)
                distance_px_block = distance_pixels[row_slice, col_slice].astype(np.float32, copy=False)
                if max_distance_pixels is None:
                    within_threshold = np.ones(distance_px_block.shape, dtype=bool)
                else:
                    within_threshold = distance_px_block < float(max_distance_pixels)
                decayed_block = grown_block - np.where(
                    anchor_block,
                    0.0,
                    distance_px_block * pixel_size_m * float(decay_frac),
                )
                grown_valid = within_threshold & np.isfinite(decayed_block) & valid_block & (decayed_block > dem_block)
                source_mask[row_slice, col_slice] = (anchor_block | grown_valid).astype(np.uint8, copy=False)

        labels = _open_temp_memmap(tmp_dir, "labels", fine_shape, np.int32)
        num_features = scipy.ndimage.label(
            source_mask,
            structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
            output=labels,
        )
        if int(num_features) <= 0:
            raise AssertionError("connected-component filtering found no grown wet regions")
        connected_labels = np.unique(labels[anchor_mask > 0])
        connected_labels = connected_labels[connected_labels > 0]

        wet_final_count = 0
        with rasterio.open(dem_fine_fp) as dem_ds, rasterio.open(output_path, "w", **output_profile) as dst_ds:
            for _, window in iter_block_windows(dst_ds, show_progress=show_progress, desc="costgrow final write pass"):
                row_slice, col_slice = _window_slices(window)
                dem_block = dem_ds.read(1, window=window).astype(np.float32, copy=False)
                valid_block = dem_valid_mask[row_slice, col_slice] > 0
                anchor_block = anchor_mask[row_slice, col_slice] > 0
                partial_block = np.array(wse_partials[row_slice, col_slice], dtype=np.float32, copy=True)
                grown_block = np.array(grown_wse[row_slice, col_slice], dtype=np.float32, copy=True)
                distance_px_block = distance_pixels[row_slice, col_slice].astype(np.float32, copy=False)
                if max_distance_pixels is None:
                    within_threshold = np.ones(distance_px_block.shape, dtype=bool)
                else:
                    within_threshold = distance_px_block < float(max_distance_pixels)
                decayed_block = grown_block - np.where(
                    anchor_block,
                    0.0,
                    distance_px_block * pixel_size_m * float(decay_frac),
                )
                grown_valid = within_threshold & np.isfinite(decayed_block) & valid_block & (decayed_block > dem_block)
                final_wse_block = np.where(
                    anchor_block,
                    partial_block,
                    np.where(grown_valid, decayed_block, np.nan),
                ).astype(np.float32, copy=False)
                labels_block = np.array(labels[row_slice, col_slice], dtype=np.int32, copy=True)
                connected_block = np.isin(labels_block, connected_labels)
                final_depth_block = np.where(
                    connected_block & valid_block,
                    np.clip(final_wse_block - dem_block, 0.0, None),
                    np.nan,
                ).astype(np.float32, copy=False)
                wet_final_count += int(np.isfinite(final_wse_block[connected_block & valid_block]).sum())
                final_depth_written = np.where(
                    np.isfinite(final_depth_block),
                    final_depth_block,
                    float(out_nodata),
                ).astype(np.float32, copy=False)
                dst_ds.write(final_depth_written, 1, window=window)

        meta = {
            "downscale": downscale,
            "dp_coarse_pixel_max": None if dp_coarse_pixel_max is None else int(dp_coarse_pixel_max),
            "decay_frac": float(decay_frac),
            "distance_fill_method": str(distance_fill_method),
            "wet_anchors": int(wet_anchor_count),
            "wet_final": int(wet_final_count),
        }
        return output_path, meta
    finally:
        _close_temp_memmap(labels)
        _close_temp_memmap(source_mask)
        _close_temp_memmap(distance_pixels)
        _close_temp_memmap(grown_wse)
        _close_temp_memmap(anchor_mask)
        _close_temp_memmap(cost)
        _close_temp_memmap(wse_partials)
        _close_temp_memmap(filled_fine_wse)
        _close_temp_memmap(wse_fine)
        _close_temp_memmap(dem_valid_mask)
        tmp_dir_obj.cleanup()


def _run_costgrow_core(
    pcraster_module,
    depth_lr: np.ndarray,
    depth_lr_profile: dict,
    dem_lr: np.ndarray,
    dem_lr_valid_mask: np.ndarray,
    dem_fine: np.ndarray,
    dem_fine_valid_mask: np.ndarray,
    fine_profile: dict,
    min_depth_threshold: float,
    dp_coarse_pixel_max: int | None,
    decay_frac: float,
    distance_fill_method: str,
    distance_fill_kwargs: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the CostGrow terrain-penalty pipeline in array space."""
    if distance_fill_kwargs is None:
        distance_fill_kwargs = {}

    coarse_wet = dem_lr_valid_mask & (depth_lr > float(min_depth_threshold))
    if not coarse_wet.any():
        raise AssertionError("depth_lr contains no wet/source cells above the minimum depth threshold")
    coarse_wse = np.where(coarse_wet, dem_lr + depth_lr, np.nan).astype(np.float32, copy=False)

    wse_fine = _resample_array_to_profile(
        coarse_wse,
        depth_lr_profile,
        fine_profile,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    wet_above_ground = np.isfinite(wse_fine) & dem_fine_valid_mask & (wse_fine > dem_fine)
    if not wet_above_ground.any():
        raise AssertionError("wet-above-ground filtering fully masked the input wet cells")
    wse_partials = np.where(wet_above_ground, wse_fine, np.nan).astype(np.float32, copy=False)

    filled_coarse_wse = _fill_nearest_unmasked(
        np.ma.MaskedArray(coarse_wse, mask=~np.isfinite(coarse_wse)),
        method=distance_fill_method,
        **distance_fill_kwargs,
    ).astype(np.float32, copy=False)
    filled_coarse_wse = np.where(np.isfinite(coarse_wse), coarse_wse, filled_coarse_wse)
    filled_fine_wse = _resample_array_to_profile(
        filled_coarse_wse,
        depth_lr_profile,
        fine_profile,
        resampling=Resampling.bilinear,
        src_nodata=None,
        dst_nodata=np.nan,
    )

    cost, _ = _compute_cost_surface(filled_fine_wse, dem_fine, dem_fine_valid_mask)
    grown_wse = _distance_fill_cost_pcraster(pcraster_module, wse_partials, cost, fine_profile)

    pixel_size_m = float(np.mean([abs(float(fine_profile["transform"].a)), abs(float(fine_profile["transform"].e))]))
    downscale = int(round(dem_fine.shape[0] / depth_lr.shape[0]))
    distance = scipy.ndimage.distance_transform_cdt(
        ~np.isfinite(wse_partials),
        return_distances=True,
        return_indices=False,
        **distance_fill_kwargs,
    ).astype(np.float32, copy=False) * pixel_size_m

    if dp_coarse_pixel_max is None:
        grow_threshold = np.ones(distance.shape, dtype=bool)
    else:
        grow_threshold = (distance / pixel_size_m / max(downscale, 1)) < int(dp_coarse_pixel_max)

    decayed_wse = grown_wse - np.where(np.isfinite(wse_partials), 0.0, distance * float(decay_frac))
    grown_valid = grow_threshold & np.isfinite(decayed_wse) & dem_fine_valid_mask & (decayed_wse > dem_fine)
    wse_after_growth = np.where(np.isfinite(wse_partials), wse_partials, np.where(grown_valid, decayed_wse, np.nan))

    connected = _filter_isolated(np.isfinite(wse_after_growth), np.isfinite(wse_partials))
    final_wse = np.where(connected & dem_fine_valid_mask, wse_after_growth, np.nan).astype(np.float32, copy=False)
    final_depth = np.where(np.isfinite(final_wse), np.clip(final_wse - dem_fine, 0.0, None), np.nan).astype(np.float32, copy=False)

    meta = {
        "downscale": downscale,
        "dp_coarse_pixel_max": None if dp_coarse_pixel_max is None else int(dp_coarse_pixel_max),
        "decay_frac": float(decay_frac),
        "distance_fill_method": str(distance_fill_method),
        "wet_anchors": int(np.isfinite(wse_partials).sum()),
        "wet_final": int(np.isfinite(final_wse).sum()),
    }
    return final_depth, meta


class ModelWorker(Model):
    """Built-in worker implementing the CostGrow terrain-penalty flow."""

    model_version = "CostGrow_Terrain"
    requires_model_artifact = False
    windowed_io_min_bytes = 32 * 1024 * 1024

    def __init__(self, model_fp: str | Path | None = None, logger=None):
        """Initialize a built-in worker that does not consume weight files."""
        super().__init__(model_fp=model_fp, model_version=self.model_version, logger=logger)

    def _resolve_execution_path(self, window_method: str, dem_fine_shape: tuple[int, int]) -> str:
        """Choose the in-memory or disk-backed CostGrow execution path."""
        fine_bytes = int(dem_fine_shape[0]) * int(dem_fine_shape[1]) * 4
        if window_method == "hard" and fine_bytes >= int(self.windowed_io_min_bytes):
            return "windowed"
        return "simple"

    def run(
        self,
        depth_lr_fp: str | Path,
        dem_hr_fp: str | Path,
        output_fp: str | Path,
        min_depth_threshold: float | None = None,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the CostGrow terrain-penalty algorithm on aligned prepared rasters."""
        start = time.perf_counter()
        log = self.log
        pcraster_module = _check_pcraster()
        window_method = str(kwargs.get("window_method", "hard") or "hard").strip().lower()
        assert window_method in {"hard", "feather"}, f"unsupported window_method={window_method}"
        assert isinstance(show_progress, bool), f"show_progress must be bool, got {type(show_progress)!r}"

        depth_lr_arr, depth_lr_nodata, depth_lr_profile = _read_single_band_raster(depth_lr_fp)
        with rasterio.open(dem_hr_fp) as dem_meta_ds:
            dem_fine_nodata = dem_meta_ds.nodata
            dem_fine_profile = dem_meta_ds.profile.copy()
            dem_fine_shape = (int(dem_meta_ds.height), int(dem_meta_ds.width))
        assert depth_lr_nodata is None, f"prepared depth_lr nodata must be None; got {depth_lr_nodata}"
        assert np.isfinite(depth_lr_arr).all(), "prepared depth_lr must contain only finite values"
        assert float(depth_lr_arr.min()) >= 0.0, f"prepared depth_lr must be >= 0; got min={float(depth_lr_arr.min())}"
        min_depth_threshold_value = 1e-3 if min_depth_threshold is None else float(min_depth_threshold)
        execution_path = self._resolve_execution_path(window_method, dem_fine_shape)
        log.info(
            "costgrow execution path\n"
            f"  window_method={window_method}\n"
            f"  execution_path={execution_path}\n"
            "  windowed_contract="
            f"{'transitional_disk_backed_global' if execution_path == 'windowed' else 'whole_scene'}"
        )
        out_nodata = dem_fine_nodata if dem_fine_nodata is not None else -9999.0
        if execution_path == "simple":
            dem_fine_arr, _, _ = _read_single_band_raster(dem_hr_fp)
            dem_fine_valid_mask = valid_mask_from_array(dem_fine_arr, dem_fine_nodata)
            dem_lr_profile = depth_lr_profile.copy()
            dem_lr_profile.update(height=int(depth_lr_arr.shape[0]), width=int(depth_lr_arr.shape[1]))
            dem_lr_arr = _resample_array_to_profile(
                dem_fine_arr,
                dem_fine_profile,
                dem_lr_profile,
                resampling=Resampling.bilinear,
                src_nodata=dem_fine_nodata,
                dst_nodata=np.nan,
            )
            dem_lr_valid_arr = _resample_array_to_profile(
                dem_fine_valid_mask.astype(np.float32, copy=False),
                dem_fine_profile,
                dem_lr_profile,
                resampling=Resampling.nearest,
                src_nodata=None,
                dst_nodata=0.0,
            )
            dem_lr_valid_mask = dem_lr_valid_arr > 0.5

            final_depth, meta = _run_costgrow_core(
                pcraster_module,
                depth_lr_arr,
                depth_lr_profile,
                dem_lr_arr,
                dem_lr_valid_mask,
                dem_fine_arr,
                dem_fine_valid_mask,
                dem_fine_profile,
                min_depth_threshold=min_depth_threshold_value,
                # TODO(issue #47): route these CostGrow-only knobs through explicit model kwargs, not shared worker args.
                dp_coarse_pixel_max=kwargs.get("dp_coarse_pixel_max", 10),
                decay_frac=float(kwargs.get("decay_frac", 0.001)),
                distance_fill_method=str(kwargs.get("distance_fill_method", "distance_transform_cdt")),
                distance_fill_kwargs=kwargs.get("distance_fill_kwargs"),
            )
            final_depth_written = np.where(np.isfinite(final_depth), final_depth, float(out_nodata)).astype(
                np.float32,
                copy=False,
            )
            out_profile = dem_fine_profile.copy()
            out_profile["nodata"] = float(out_nodata)
            out_fp = _write_single_band_raster(output_fp, final_depth_written, out_profile)
        else:
            out_fp, meta = _run_costgrow_core_windowed(
                pcraster_module,
                depth_lr_arr,
                depth_lr_profile,
                dem_hr_fp,
                dem_fine_nodata,
                dem_fine_profile,
                output_fp,
                float(out_nodata),
                min_depth_threshold=min_depth_threshold_value,
                dp_coarse_pixel_max=kwargs.get("dp_coarse_pixel_max", 10),
                decay_frac=float(kwargs.get("decay_frac", 0.001)),
                distance_fill_method=str(kwargs.get("distance_fill_method", "distance_transform_cdt")),
                distance_fill_kwargs=kwargs.get("distance_fill_kwargs"),
                show_progress=show_progress,
            )

        runtime_s = time.perf_counter() - start
        out_size = int(out_fp.stat().st_size)
        log.info(f"finished CostGrow terrain penalty in {runtime_s:.3f}s; wrote {out_size:,} bytes to\n    {out_fp}")
        return {
            "output_fp": str(out_fp),
            "runtime_s": float(runtime_s),
            "model_version": self.model_version,
            "model_fp": None,
            "output_size_bytes": out_size,
            "execution_path": execution_path,
            "preprocess": {
                "min_depth_threshold": float(min_depth_threshold_value),
                "window_method": window_method,
                "prepared_inputs": {
                    "depth_lr_prepared_fp": str(Path(depth_lr_fp).expanduser().resolve()),
                    "dem_hr_prepared_fp": str(Path(dem_hr_fp).expanduser().resolve()),
                    "depth_lr_nodata": depth_lr_nodata,
                    "dem_hr_nodata": dem_fine_nodata,
                },
                "costgrow": {
                    **meta,
                    "windowed_contract": (
                        "transitional_disk_backed_global" if execution_path == "windowed" else "whole_scene"
                    ),
                },
            },
        }
