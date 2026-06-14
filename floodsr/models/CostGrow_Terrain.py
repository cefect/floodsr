"""Built-in CostGrow terrain-penalty worker."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import scipy.ndimage
from rasterio.transform import array_bounds
from rasterio.windows import Window
from rasterio.warp import Resampling, reproject

from floodsr.engine.pcraster_check import _check_pcraster
from floodsr.models.base import Model
from floodsr.preprocessing import (
    _build_single_band_profile,
    _read_single_band_raster,
    _write_single_band_raster,
    valid_mask_from_array,
)
from floodsr.tiling import build_tile_starts, iter_block_windows, iter_window_origins


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


def _expand_window(window: Window, halo_pixels: int, max_shape: tuple[int, int]) -> Window:
    """Expand one core window by a symmetric halo clipped to raster bounds."""
    halo = max(int(halo_pixels), 0)
    row_off = max(int(window.row_off) - halo, 0)
    col_off = max(int(window.col_off) - halo, 0)
    row_stop = min(int(window.row_off + window.height) + halo, int(max_shape[0]))
    col_stop = min(int(window.col_off + window.width) + halo, int(max_shape[1]))
    return Window(col_off=col_off, row_off=row_off, width=col_stop - col_off, height=row_stop - row_off)


def _crop_from_padded_window(core_window: Window, padded_window: Window) -> tuple[slice, slice]:
    """Return row/col crop slices that extract the core tile from a padded tile array."""
    row0 = int(core_window.row_off - padded_window.row_off)
    col0 = int(core_window.col_off - padded_window.col_off)
    return (
        slice(row0, row0 + int(core_window.height)),
        slice(col0, col0 + int(core_window.width)),
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
    tile_core_size_px: int,
    tile_halo_pixels: int,
) -> tuple[Path, dict[str, Any]]:
    """Run CostGrow with hard-window bounded-region tiles plus halo context."""
    if distance_fill_kwargs is None:
        distance_fill_kwargs = {}
    if dp_coarse_pixel_max is None:
        raise AssertionError("windowed CostGrow requires finite dp_coarse_pixel_max to bound tile halos")

    # Initialize the output raster and the coarse-grid DEM support arrays once.
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
    dem_lr_arr = np.full(depth_lr.shape, np.nan, dtype=np.float32)
    dem_lr_valid_arr = np.zeros(depth_lr.shape, dtype=np.uint8)

    # Reproject the fine DEM down to the coarse grid used to reconstruct coarse WSE anchors.
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

    # Build coarse validity conservatively by aggregating fine valid pixels block by block.
    with rasterio.open(dem_fine_fp) as dem_ds:
        for _, window in iter_block_windows(dem_ds, show_progress=show_progress, desc="costgrow coarse mask pass"):
            dem_block = dem_ds.read(1, window=window).astype(np.float32, copy=False)
            valid_block = valid_mask_from_array(dem_block, dem_fine_nodata).astype(np.uint8, copy=False)
            block_lr_valid = np.zeros(depth_lr.shape, dtype=np.uint8)
            reproject(
                source=valid_block,
                destination=block_lr_valid,
                src_transform=dem_ds.window_transform(window),
                src_crs=dem_ds.crs,
                dst_transform=depth_lr_profile["transform"],
                dst_crs=depth_lr_profile["crs"],
                src_nodata=0,
                dst_nodata=0,
                resampling=Resampling.nearest,
                num_threads=1,
            )
            np.maximum(dem_lr_valid_arr, block_lr_valid, out=dem_lr_valid_arr)
    dem_lr_valid_mask = dem_lr_valid_arr > 0

    # Reconstruct coarse wet cells and a filled coarse WSE field shared by all tiles.
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

    pixel_size_m = float(np.mean([abs(float(fine_profile["transform"].a)), abs(float(fine_profile["transform"].e))]))
    downscale = int(round(fine_shape[0] / depth_lr.shape[0]))
    max_distance_pixels = int(dp_coarse_pixel_max) * max(downscale, 1)
    wet_anchor_count = 0
    wet_final_count = 0

    tile_h = min(int(tile_core_size_px), fine_shape[0])
    tile_w = min(int(tile_core_size_px), fine_shape[1])
    y_starts = build_tile_starts(fine_shape[0], tile_h, tile_h)
    x_starts = build_tile_starts(fine_shape[1], tile_w, tile_w)

    # Process one padded tile at a time, then write only the cropped core to the output.
    with rasterio.open(dem_fine_fp) as dem_ds, rasterio.open(output_path, "w", **output_profile) as dst_ds:
        for _, _, y0, x0 in iter_window_origins(
            y_starts,
            x_starts,
            show_progress=show_progress,
            desc="costgrow tile pass",
        ):
            core_window = Window(
                col_off=int(x0),
                row_off=int(y0),
                width=min(tile_w, fine_shape[1] - int(x0)),
                height=min(tile_h, fine_shape[0] - int(y0)),
            )
            padded_window = _expand_window(core_window, tile_halo_pixels, fine_shape)
            core_crop = _crop_from_padded_window(core_window, padded_window)

            dem_tile = dem_ds.read(1, window=padded_window).astype(np.float32, copy=False)
            valid_tile = valid_mask_from_array(dem_tile, dem_fine_nodata)
            tile_profile = fine_profile.copy()
            tile_profile.update(
                height=int(padded_window.height),
                width=int(padded_window.width),
                transform=dem_ds.window_transform(padded_window),
            )

            # Interpolate the coarse anchor/fill fields onto this padded fine-grid tile.
            wse_tile = _resample_array_to_profile(
                coarse_wse,
                depth_lr_profile,
                tile_profile,
                resampling=Resampling.bilinear,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
            filled_tile = _resample_array_to_profile(
                filled_coarse_wse,
                depth_lr_profile,
                tile_profile,
                resampling=Resampling.bilinear,
                src_nodata=None,
                dst_nodata=np.nan,
            )
            anchor_tile = np.isfinite(wse_tile) & valid_tile & (wse_tile > dem_tile)
            if not anchor_tile.any():
                empty_core = np.full(
                    (int(core_window.height), int(core_window.width)),
                    float(out_nodata),
                    dtype=np.float32,
                )
                dst_ds.write(empty_core, 1, window=core_window)
                continue

            # Run the CostGrow growth/decay/connectivity sequence within the padded tile only.
            partial_tile = np.where(anchor_tile, wse_tile, np.nan).astype(np.float32, copy=False)
            cost_tile, _ = _compute_cost_surface(filled_tile, dem_tile, valid_tile)
            grown_tile = _distance_fill_cost_pcraster(pcraster_module, partial_tile, cost_tile, tile_profile)
            distance_px_tile = scipy.ndimage.distance_transform_cdt(
                ~anchor_tile,
                return_distances=True,
                return_indices=False,
                **distance_fill_kwargs,
            ).astype(np.float32, copy=False)
            within_threshold = distance_px_tile < float(max_distance_pixels)
            decayed_tile = grown_tile - np.where(anchor_tile, 0.0, distance_px_tile * pixel_size_m * float(decay_frac))
            grown_valid = within_threshold & np.isfinite(decayed_tile) & valid_tile & (decayed_tile > dem_tile)
            final_wse_tile = np.where(anchor_tile, partial_tile, np.where(grown_valid, decayed_tile, np.nan)).astype(
                np.float32,
                copy=False,
            )
            connected_tile = _filter_isolated(np.isfinite(final_wse_tile), anchor_tile)
            final_depth_tile = np.where(
                connected_tile & valid_tile,
                np.clip(final_wse_tile - dem_tile, 0.0, None),
                np.nan,
            ).astype(np.float32, copy=False)

            # Crop back to the core window so overlap/halo context never leaks into neighbors.
            core_anchor = anchor_tile[core_crop]
            core_final_depth = final_depth_tile[core_crop]
            wet_anchor_count += int(core_anchor.sum())
            wet_final_count += int(np.isfinite(core_final_depth).sum())
            core_written = np.where(
                np.isfinite(core_final_depth),
                core_final_depth,
                float(out_nodata),
            ).astype(np.float32, copy=False)
            dst_ds.write(core_written, 1, window=core_window)

    meta = {
        "downscale": downscale,
        "dp_coarse_pixel_max": int(dp_coarse_pixel_max),
        "decay_frac": float(decay_frac),
        "distance_fill_method": str(distance_fill_method),
        "wet_anchors": int(wet_anchor_count),
        "wet_final": int(wet_final_count),
        "tile_contract_mode": "tile_halo",
        "tile_core_size_px": int(tile_core_size_px),
        "tile_halo_px": int(tile_halo_pixels),
        "tile_anchor_radius_px": int(max_distance_pixels),
        "merge_rule": "hard_crop_core",
        "staged_state": "global_coarse_prefill_plus_tile_local_recompute",
    }
    return output_path, meta


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

    # Reconstruct coarse wet anchors and coarse WSE from prepared depth and DEM inputs.
    coarse_wet = dem_lr_valid_mask & (depth_lr > float(min_depth_threshold))
    if not coarse_wet.any():
        raise AssertionError("depth_lr contains no wet/source cells above the minimum depth threshold")
    coarse_wse = np.where(coarse_wet, dem_lr + depth_lr, np.nan).astype(np.float32, copy=False)

    # Interpolate the coarse anchors onto the fine grid and keep only wet-above-ground partials.
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

    # Fill the coarse WSE holes first, then use that filled field to derive the terrain cost surface.
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

    # Limit growth by coarse-pixel distance, then apply decay away from the original anchors.
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

    # Remove disconnected grown islands, then convert the final WSE surface back to depth.
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
    windowed_core_tile_size_px = 2048
    windowed_halo_factor = 2

    def __init__(self, model_fp: str | Path | None = None, logger=None):
        """Initialize a built-in worker that does not consume weight files."""
        super().__init__(model_fp=model_fp, model_version=self.model_version, logger=logger)

    def _resolve_execution_path(self, window_method: str, dem_fine_shape: tuple[int, int]) -> str:
        """Choose the in-memory or disk-backed CostGrow execution path."""
        fine_bytes = int(dem_fine_shape[0]) * int(dem_fine_shape[1]) * 4
        if window_method == "hard" and fine_bytes >= int(self.windowed_io_min_bytes):
            return "windowed"
        return "simple"

    def _resolve_windowed_tile_contract(self, downscale: int, dp_coarse_pixel_max: int | None) -> dict[str, int | str]:
        """Build the explicit CostGrow hard-window tile contract for bounded-region execution."""
        if dp_coarse_pixel_max is None:
            raise AssertionError("windowed CostGrow requires finite dp_coarse_pixel_max to define a bounded halo")
        if int(dp_coarse_pixel_max) < 0:
            raise AssertionError(f"dp_coarse_pixel_max must be >= 0; got {dp_coarse_pixel_max}")
        anchor_radius_px = int(dp_coarse_pixel_max) * max(int(downscale), 1)
        halo_px = max(anchor_radius_px * int(self.windowed_halo_factor), max(int(downscale), 1))
        return {
            "mode": "tile_halo",
            "core_tile_size_px": int(self.windowed_core_tile_size_px),
            "halo_px": int(halo_px),
            "anchor_radius_px": int(anchor_radius_px),
            "merge_rule": "hard_crop_core",
            "staged_state": "global_coarse_prefill_plus_tile_local_recompute",
        }

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

        # Validate the runtime mode and prepared inputs before choosing a solve path.
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
        dp_coarse_pixel_max_value = kwargs.get("dp_coarse_pixel_max", 10)
        downscale = int(round(dem_fine_shape[0] / depth_lr_arr.shape[0]))
        execution_path = self._resolve_execution_path(window_method, dem_fine_shape)
        tile_contract = None
        if execution_path == "windowed":
            tile_contract = self._resolve_windowed_tile_contract(downscale, dp_coarse_pixel_max_value)
        log.info(
            "costgrow execution path\n"
            f"  window_method={window_method}\n"
            f"  execution_path={execution_path}\n"
            "  windowed_contract="
            f"{tile_contract['mode'] if tile_contract is not None else 'whole_scene'}"
        )
        out_nodata = dem_fine_nodata if dem_fine_nodata is not None else -9999.0
        if execution_path == "simple":
            # The simple path materializes the full fine-grid solve in memory.
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
                dp_coarse_pixel_max=dp_coarse_pixel_max_value,
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
            # The windowed path keeps the coarse support global but bounds fine-grid work per tile.
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
                dp_coarse_pixel_max=dp_coarse_pixel_max_value,
                decay_frac=float(kwargs.get("decay_frac", 0.001)),
                distance_fill_method=str(kwargs.get("distance_fill_method", "distance_transform_cdt")),
                distance_fill_kwargs=kwargs.get("distance_fill_kwargs"),
                show_progress=show_progress,
                tile_core_size_px=int(tile_contract["core_tile_size_px"]),
                tile_halo_pixels=int(tile_contract["halo_px"]),
            )

        # Return one structured runtime payload so callers can inspect the chosen execution mode.
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
                    "windowed_contract": tile_contract["mode"] if tile_contract is not None else "whole_scene",
                },
            },
        }
