"""16x DEM-conditioned ResUNet 

 

Architecture Description:
This model performs 16x single-channel depth super-resolution with DEM
conditioning. It consumes two inputs:
- `depth_lr`: low-resolution depth chip, default shape `(32, 32, 1)`.
- `dem_hr`: high-resolution DEM chip aligned to the target grid, default shape
  `(512, 512, 1)` for 16x.

Depth chips are clipped to `[0, max_depth]`, transformed with `log1p`, and
scaled to `[0, 1]`. DEM chips are clipped at a robust upper percentile
(`dem_pct_clip`) and min-max normalized per chip. The architecture is a
dual-scale, DEM-aware ResUNet:
- `dem_hr` is average-pooled to LR (`dem_lr`) and concatenated with `depth_lr`
  as encoder input.
- The encoder/decoder backbone is a 4-level UNet with residual blocks at each
  scale; channel widths are `f, 2f, 4f, 8f, 16f` (`f=base_filters`).
- After decoder reconstruction at LR, a transposed convolution upsamples by 16x
  to HR.
- The upsampled feature map is fused again with `dem_hr` before the final
  1-channel linear prediction head (`depth_hr_pred`), so topography informs both
  coarse and fine prediction stages.

 

Training run summary:
Training uses Adam with `clipnorm=1.0` and a piecewise-constant learning-rate
schedule (`1e-4` then `5e-5` halfway through total steps). Loss is MAE, with
metrics `PSNR`, `SSIM`, `RMSE`, `RMSE_wet`, and `CSI`. The train pipeline uses
deterministic index splitting, optional tf.data cache, optional flip/rot90
augmentation on training only, repeat+batch+prefetch, and configurable
`steps_per_epoch`.

Inference:
1. Model specific pre-processing
- Load `train_config.json` and resolve model parameters (`SCALE`, LR/HR tile geometry, `MAX_DEPTH`, DEM clip settings).
- Validate input raster compatibility (CRS, bounds, and grid checks).
- Keep LR depth on raw LR grid.
- Resample HR depth and DEM to model-space HR grid derived from `raw_lr_shape * SCALE`.
- Apply depth normalization using `log1p(clip(depth, 0, MAX_DEPTH)) / log1p(MAX_DEPTH)`.
- Keep DEM normalization as tile-local (computed inside the inference loop), matching notebook behavior.

2. Tiling/windowing
- Pad model-space arrays so LR/HR windows align exactly with fixed model tile sizes.
- Build non-overlap HR window origins and map each HR origin to LR origin by integer `SCALE`.
- Build feathered overlap window grid with fixed overlap/stride and forced trailing-edge coverage.
- Reuse cached tile predictions by `(y0, x0)` key to avoid duplicate model calls across passes.

3. Core inference at model-engine boundary
- For each window, slice aligned LR depth and HR DEM tiles.
- Normalize LR/DEM inputs to `[0, 1]` using tile-local DEM stats.
- Expand to batched NHWC tensors and execute model forward pass at the boundary contract.
- Validate/persist per-tile prediction outputs and cache them for downstream stitching/diagnostics.

4. Mosaicking/stitching
- Run an initial non-overlap chip pass to populate chip outputs and diagnostics arrays.
- Run feathered mosaicking pass over overlap windows using separable 1D feather ramps.
- Flatten boundary feather weights on scene edges to avoid dimming at domain boundaries.
- Accumulate weighted predictions and normalize by accumulated weight sum.
- Crop stitched output back to valid model-space extent.

5. Model specific post-processing
- Convert stitched SR output to depth meters and clamp depth range.
- Resample model-space SR depth back to raw HR grid (post-resample step).
- Apply low-depth mask in meter domain.
- Re-normalize to `[0, 1]` where needed for metric helper compatibility.
- Compute/export full-scene diagnostics (including bilinear baseline comparison) and write output when enabled.

 
"""

import logging, math, shutil, tempfile, time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from tqdm import tqdm

from floodsr.engine import EngineORT
from floodsr.models.base import Model
from floodsr.preprocessing import _build_single_band_profile, _read_single_band_raster, _write_single_band_raster, _zero_nodata_in_place, replace_nodata_with_zero, resolve_preprocess_config
from floodsr.tiling import build_feather_ramp, build_tile_starts, iter_block_windows, iter_window_origins

from rasterio.transform import from_bounds as bounds_to_transform
from rasterio.windows import Window
from rasterio.warp import Resampling, reproject


try:
    from osgeo import gdal
except ImportError:
    gdal = None


def _pixel_size_m(profile: dict) -> tuple[float, float]:
    """Extract absolute pixel size in projection units."""
    transform = profile.get("transform")
    if transform is None:
        return (float("nan"), float("nan"))
    if hasattr(transform, "a") and hasattr(transform, "e"):
        return (abs(float(transform.a)), abs(float(transform.e)))
    return (abs(float(transform[0])), abs(float(transform[4])))


def _profile_bounds(profile: dict) -> tuple[float, float, float, float]:
    """Compute raster bounds from profile height/width/transform."""
    from rasterio.transform import array_bounds

    height = int(profile.get("height"))
    width = int(profile.get("width"))
    transform = profile.get("transform")
    assert height > 0 and width > 0, f"profile height/width must be > 0; got {(height, width)}"
    assert transform is not None, "profile transform is required to compute bounds"
    left, bottom, right, top = array_bounds(height, width, transform)
    return (float(left), float(bottom), float(right), float(top))


class ModelWorker(Model):
    """Model worker implementing notebook-parity ToHR flow for version `ResUNet_16x_DEM`."""

    model_version = "ResUNet_16x_DEM"
    low_depth_mask_m = 1e-3
    windowed_io_min_bytes = 32 * 1024 * 1024

    def __init__(
        self,
        model_fp: str | Path,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
        logger=None,
    ):
        """Initialize worker state and provider policy."""
        super().__init__(model_fp=model_fp, model_version=self.model_version, logger=logger)
        assert providers, "providers cannot be empty"
        self.providers = tuple(providers)
        self.engine: EngineORT | None = None

    def __enter__(self):
        """Create runtime resources for this worker context."""
        self.engine = EngineORT(self.model_fp, providers=self.providers, logger=self.log)
        return self

    def __exit__(self, exc_type, exc, tb):
        """Release runtime resources when context exits."""
        if self.engine is not None and hasattr(self.engine, "close"):
            self.engine.close()
        self.engine = None
        return False

    def _predict_tile_depth_m(
        self,
        depth_tile: np.ndarray,
        dem_tile: np.ndarray,
        max_depth: float,
        dem_pct_clip: float,
        tile_dem_stats_l: list[dict[str, float]],
    ) -> np.ndarray:
        """Run one model tile and collect DEM normalization stats."""
        assert self.engine is not None, "worker must be entered before running inference"
        log = self.log
        run_result = self.engine.run_tile(
            depth_tile,
            dem_tile,
            max_depth=max_depth,
            dem_pct_clip=dem_pct_clip,
            dem_ref_stats=None,
            normalize_inputs=True,
            depth_lr_nodata=None,
            dem_hr_nodata=None,
            logger=log,
        )
        pred_depth_m = run_result["prediction_m"]
        dem_stats_used = run_result.get("dem_stats_used")
        if isinstance(dem_stats_used, dict):
            tile_dem_stats_l.append(
                {
                    "dem_p_clip": float(dem_stats_used.get("p_clip", 0.0)),
                    "dem_min": float(dem_stats_used.get("dem_min", 0.0)),
                    "dem_max": float(dem_stats_used.get("dem_max", 0.0)),
                }
            )
        return pred_depth_m

    def _summarize_tile_dem_stats(
        self,
        tile_dem_stats_l: list[dict[str, float]],
    ) -> dict[str, float] | None:
        """Summarize tile-local DEM normalization stats."""
        if not tile_dem_stats_l:
            return None
        dem_stats_np = np.asarray(
            [
                [
                    float(meta["dem_p_clip"]),
                    float(meta["dem_min"]),
                    float(meta["dem_max"]),
                ]
                for meta in tile_dem_stats_l
            ],
            dtype=np.float32,
        )
        dem_range_np = dem_stats_np[:, 2] - dem_stats_np[:, 1]
        return {
            "tile_count": float(dem_stats_np.shape[0]),
            "dem_p_clip_min": float(dem_stats_np[:, 0].min()),
            "dem_p_clip_mean": float(dem_stats_np[:, 0].mean()),
            "dem_p_clip_max": float(dem_stats_np[:, 0].max()),
            "dem_range_min": float(dem_range_np.min()),
            "dem_range_mean": float(dem_range_np.mean()),
            "dem_range_max": float(dem_range_np.max()),
        }

    def _resolve_execution_path(self, window_method: str, dem_raw_shape: tuple[int, int]) -> str:
        """Choose the in-memory or raster-backed execution path."""
        raw_bytes = int(dem_raw_shape[0]) * int(dem_raw_shape[1]) * 4
        if window_method == "hard" and raw_bytes >= int(self.windowed_io_min_bytes):
            return "windowed"
        return "simple"

    def _postprocess_output_in_place(self, output_fp: str | Path, max_depth: float) -> None:
        """Apply clipping and low-depth masking to an on-disk raster."""
        out_path = Path(output_fp).expanduser().resolve()
        with rasterio.open(out_path, "r+") as ds:
            for _, window in ds.block_windows(1):
                arr = ds.read(1, window=window).astype(np.float32, copy=False)
                arr = np.clip(arr, 0.0, float(max_depth)).astype(np.float32, copy=False)
                arr = np.where(arr < float(self.low_depth_mask_m), 0.0, arr).astype(np.float32, copy=False)
                ds.write(arr, 1, window=window)

    def _gdal_is_available(self) -> bool:
        """Return True when GDAL Python bindings are available."""
        return gdal is not None

    def _build_windowed_output_vrt(self, output_fp: str | Path, tile_fp_l: list[Path], nodata: float | None) -> Path:
        """Build one simple VRT over one or more windowed output tiles."""
        out_path = Path(output_fp).expanduser().resolve()
        vrt_fp = out_path.with_suffix(".vrt")
        if vrt_fp.exists():
            vrt_fp.unlink()
        vrt_options = gdal.BuildVRTOptions(
            srcNodata=float(nodata) if nodata is not None else None,
            VRTNodata=float(nodata) if nodata is not None else None,
        )
        vrt_ds = gdal.BuildVRT(str(vrt_fp), [str(fp) for fp in tile_fp_l], options=vrt_options)
        assert vrt_ds is not None, f"gdal.BuildVRT returned None for {vrt_fp}"
        vrt_ds = None
        return vrt_fp

    def _write_prediction_array(
        self,
        output_fp: str | Path,
        prediction_out_m: np.ndarray,
        output_profile: dict,
        max_depth: float,
        show_progress: bool,
    ) -> Path:
        """Clip, mask, and write an in-memory prediction array blockwise."""
        out_path = Path(output_fp).expanduser().resolve()
        with rasterio.open(out_path, "w", **output_profile) as dst_ds:
            for _, window in iter_block_windows(
                dst_ds,
                show_progress=show_progress,
                desc="final write pass",
            ):
                row_off = int(window.row_off)
                col_off = int(window.col_off)
                height = int(window.height)
                width = int(window.width)
                arr = prediction_out_m[row_off : row_off + height, col_off : col_off + width].astype(
                    np.float32,
                    copy=False,
                )
                arr = np.clip(arr, 0.0, float(max_depth)).astype(np.float32, copy=False)
                arr = np.where(arr < float(self.low_depth_mask_m), 0.0, arr).astype(np.float32, copy=False)
                dst_ds.write(arr, 1, window=window)
        return out_path

    def _run_tiled_model_on_prepared(
        self,
        depth_lr_raw: np.ndarray,
        dem_hr_raw: np.ndarray,
        depth_lr_profile: dict,
        dem_hr_profile: dict,
        preprocess_cfg: dict[str, object],
        model_lr_tile: int,
        model_scale: int,
        contract_hr_tile: int,
        window_method: str,
        overlap_lr: int,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, int, dict[str, float] | None]:
        """
        Run tiled model execution over prepared rasters and return model-space SR in meter domain.

        Parameters
        ----------
        depth_lr_raw:
            Prepared low-resolution depth array.
        dem_hr_raw:
            Prepared high-resolution DEM array on model grid.
        depth_lr_profile:
            Profile metadata for prepared low-resolution depth.
        dem_hr_profile:
            Profile metadata for prepared high-resolution DEM.
        preprocess_cfg:
            Resolved preprocessing configuration dictionary.
        model_lr_tile:
            Low-resolution tile edge in pixels.
        model_scale:
            Integer scale ratio from LR tile grid to HR tile grid.
        contract_hr_tile:
            High-resolution tile edge in pixels from runtime contract.
        window_method:
            Mosaicing strategy (`hard` or `feather`).
        overlap_lr:
            Feather overlap in low-resolution pixels.
        show_progress:
            Whether tiled inference progress bars should be rendered.

        Returns
        -------
        tuple[np.ndarray, int, dict[str, float] | None]
            Model-space SR depth in meters, tile-cache size, and summary of tile DEM stats.
        """
        log = self.log
        assert self.engine is not None, "worker must be entered before running inference"
        assert window_method in {"hard", "feather"}, f"unsupported window_method={window_method}"

        # Validate prepared in-memory arrays before tiling and inference.
        assert depth_lr_raw.ndim == 2, f"aligned depth must be 2D; got {depth_lr_raw.shape}"
        assert dem_hr_raw.ndim == 2, f"aligned DEM must be 2D; got {dem_hr_raw.shape}"
        assert np.isfinite(depth_lr_raw).all(), "aligned depth contains non-finite values"
        assert np.isfinite(dem_hr_raw).all(), "aligned DEM contains non-finite values"

        max_depth = float(preprocess_cfg["max_depth"])
        dem_pct_clip = float(preprocess_cfg["dem_pct_clip"])
        crop_h, crop_w = dem_hr_raw.shape
        expected_lr_h = crop_h // model_scale
        expected_lr_w = crop_w // model_scale
        assert expected_lr_h > 0 and expected_lr_w > 0, (
            f"expected low-resolution shape invalid {(expected_lr_h, expected_lr_w)} from crop {(crop_h, crop_w)} "
            f"and scale={model_scale}"
        )
        assert depth_lr_raw.shape == (expected_lr_h, expected_lr_w), (
            f"depth shape {depth_lr_raw.shape} does not match crop/scale target {(expected_lr_h, expected_lr_w)}"
        )
        if float(depth_lr_raw.min()) > max_depth:
            log.warning("low-res depth values exceed max_depth; model preprocessing will clip them.")

        log.info(
            "prepared inputs summary:\n"
            f"  aligned depth_lr shape={depth_lr_raw.shape} res={_pixel_size_m(depth_lr_profile)} m/pix\n"
            f"  aligned dem_hr shape={dem_hr_raw.shape} res={_pixel_size_m(dem_hr_profile)} m/pix\n"
            f"  max_depth={max_depth}\n"
            f"  dem_pct_clip={dem_pct_clip}"
        )

        # Pad both arrays so fixed-size tiles exactly cover the model-space extent.
        pad_h = (int(math.ceil(crop_h / contract_hr_tile)) * contract_hr_tile) - crop_h
        pad_w = (int(math.ceil(crop_w / contract_hr_tile)) * contract_hr_tile) - crop_w
        dem_pad = np.pad(dem_hr_raw, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)

        hr_pad_h, hr_pad_w = dem_pad.shape
        target_depth_pad_h = hr_pad_h // model_scale
        target_depth_pad_w = hr_pad_w // model_scale
        depth_pad_extra_h = target_depth_pad_h - depth_lr_raw.shape[0]
        depth_pad_extra_w = target_depth_pad_w - depth_lr_raw.shape[1]
        assert depth_pad_extra_h >= 0 and depth_pad_extra_w >= 0, (
            f"computed LR padding must be >= 0; got {(depth_pad_extra_h, depth_pad_extra_w)}"
        )
        depth_pad = np.pad(
            depth_lr_raw,
            ((0, depth_pad_extra_h), (0, depth_pad_extra_w)),
            mode="constant",
            constant_values=0.0,
        )
        assert depth_pad.shape == (hr_pad_h // model_scale, hr_pad_w // model_scale), (
            f"depth pad shape {depth_pad.shape} incompatible with HR pad {(hr_pad_h, hr_pad_w)}"
        )

        overlap_hr = overlap_lr * model_scale
        tile_cache: dict[tuple[int, int], np.ndarray] = {}
        tile_dem_stats_l: list[dict[str, float]] = []
        log.info(
            "window config\n"
            f"  method={window_method}\n"
            f"  overlap_lr={overlap_lr}\n"
            f"  overlap_hr={overlap_hr}\n"
            f"  tile_size_lr={model_lr_tile}\n"
            f"  tile_size_hr={contract_hr_tile}"
        )

        # Cache per-tile model outputs because overlap windows revisit origins.
        def _predict_cached_tile(y0: int, x0: int) -> np.ndarray:
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

            pred_depth_m = self._predict_tile_depth_m(
                depth_tile=depth_tile,
                dem_tile=dem_tile,
                max_depth=max_depth,
                dem_pct_clip=dem_pct_clip,
                tile_dem_stats_l=tile_dem_stats_l,
            )
            assert pred_depth_m.shape == (contract_hr_tile, contract_hr_tile), (
                f"prediction shape {pred_depth_m.shape} != {(contract_hr_tile, contract_hr_tile)}"
            )
            tile_cache[key] = pred_depth_m
            return pred_depth_m

        # Route tiling by requested mosaicing method.
        if window_method == "hard":
            # Keep hard mode behavior: a single non-overlap inference sweep.
            nonoverlap_y = list(range(0, hr_pad_h, contract_hr_tile))
            nonoverlap_x = list(range(0, hr_pad_w, contract_hr_tile))
            sr_pad = np.zeros_like(dem_pad, dtype=np.float32)
            log.info(
                f"running hard tiling over {len(nonoverlap_y)}x{len(nonoverlap_x)} grid\n"
                f"  overlap_lr={overlap_lr}\n"
                f"  overlap_hr={overlap_hr}"
            )
            for _, _, y0, x0 in iter_window_origins(
                nonoverlap_y,
                nonoverlap_x,
                show_progress=show_progress,
                desc="non-overlap pass",
            ):
                pred_depth_m = _predict_cached_tile(y0, x0)
                sr_pad[y0 : y0 + contract_hr_tile, x0 : x0 + contract_hr_tile] = pred_depth_m
        elif window_method == "feather":
            # Skip hard-pass priming and run only overlap-aware feather blending.
            stride_hr = contract_hr_tile - overlap_hr
            if overlap_lr <= 0:
                raise AssertionError("feather windowing requires overlap_lr > 0")
            if stride_hr <= 0:
                raise AssertionError(
                    f"feather stride must be > 0; overlap_lr={overlap_lr}, tile={contract_hr_tile}"
                )

            y_starts = build_tile_starts(hr_pad_h, contract_hr_tile, stride_hr)
            x_starts = build_tile_starts(hr_pad_w, contract_hr_tile, stride_hr)
            feather_1d = build_feather_ramp(contract_hr_tile, overlap_hr)
            accum = np.zeros_like(dem_pad, dtype=np.float32)
            weight_sum = np.zeros_like(dem_pad, dtype=np.float32)
            log.info(
                f"running feather tiling over {len(y_starts)}x{len(x_starts)} grid\n"
                f"  stride_hr={stride_hr}\n"
                f"  overlap_hr={overlap_hr}"
            )
            for yi, xi, y0, x0 in iter_window_origins(
                y_starts,
                x_starts,
                show_progress=show_progress,
                desc="feather pass",
            ):
                pred_depth_m = _predict_cached_tile(y0, x0)
                wy = feather_1d.copy()
                wx = feather_1d.copy()
                if overlap_hr > 0:
                    if yi == 0:
                        wy[:overlap_hr] = 1.0
                    if yi == len(y_starts) - 1:
                        wy[-overlap_hr:] = 1.0
                    if xi == 0:
                        wx[:overlap_hr] = 1.0
                    if xi == len(x_starts) - 1:
                        wx[-overlap_hr:] = 1.0

                weight = np.outer(wy, wx).astype(np.float32, copy=False)
                accum[y0 : y0 + contract_hr_tile, x0 : x0 + contract_hr_tile] += pred_depth_m * weight
                weight_sum[y0 : y0 + contract_hr_tile, x0 : x0 + contract_hr_tile] += weight

            sr_pad = np.divide(
                accum,
                np.maximum(weight_sum, 1e-6),
                out=np.zeros_like(accum),
                where=weight_sum > 0,
            )
        else:  # pragma: no cover - guarded by assertions
            raise AssertionError(f"unsupported window_method={window_method}")

        prediction_depth_m = np.clip(sr_pad[:crop_h, :crop_w], 0.0, max_depth).astype(np.float32, copy=False)
        assert prediction_depth_m.ndim == 2, f"prediction must be 2D; got {prediction_depth_m.shape}"
        return prediction_depth_m, len(tile_cache), self._summarize_tile_dem_stats(tile_dem_stats_l)

    def _run_tiled_model_on_prepared_windowed_hard(
        self,
        depth_lr_fp: str | Path,
        dem_hr_fp: str | Path,
        prediction_model_fp: str | Path,
        preprocess_cfg: dict[str, object],
        model_lr_tile: int,
        model_scale: int,
        contract_hr_tile: int,
        show_progress: bool = True,
    ) -> tuple[int, dict[str, float] | None]:
        """Run hard-window inference with raster-backed reads and writes."""
        log = self.log
        depth_path = Path(depth_lr_fp).expanduser().resolve()
        dem_path = Path(dem_hr_fp).expanduser().resolve()
        pred_path = Path(prediction_model_fp).expanduser().resolve()
        max_depth = float(preprocess_cfg["max_depth"])
        dem_pct_clip = float(preprocess_cfg["dem_pct_clip"])
        tile_dem_stats_l: list[dict[str, float]] = []

        with rasterio.open(depth_path) as depth_ds, rasterio.open(dem_path) as dem_ds:
            crop_h, crop_w = int(dem_ds.height), int(dem_ds.width)
            expected_lr_shape = (crop_h // model_scale, crop_w // model_scale)
            assert (int(depth_ds.height), int(depth_ds.width)) == expected_lr_shape, (
                f"depth shape {(int(depth_ds.height), int(depth_ds.width))} does not match crop/scale target {expected_lr_shape}"
            )
            log.info(
                "prepared inputs summary:\n"
                f"  aligned depth_lr shape={expected_lr_shape} res={_pixel_size_m(depth_ds.profile)} m/pix\n"
                f"  aligned dem_hr shape={(crop_h, crop_w)} res={_pixel_size_m(dem_ds.profile)} m/pix\n"
                f"  max_depth={max_depth}\n"
                f"  dem_pct_clip={dem_pct_clip}"
            )

            pad_h = (int(math.ceil(crop_h / contract_hr_tile)) * contract_hr_tile) - crop_h
            pad_w = (int(math.ceil(crop_w / contract_hr_tile)) * contract_hr_tile) - crop_w
            hr_pad_h = crop_h + pad_h
            hr_pad_w = crop_w + pad_w
            nonoverlap_y = list(range(0, hr_pad_h, contract_hr_tile))
            nonoverlap_x = list(range(0, hr_pad_w, contract_hr_tile))
            pred_profile = _build_single_band_profile(
                pred_path,
                dem_ds.profile,
                crop_h,
                crop_w,
                dem_ds.transform,
            )
            log.info(
                "window config\n"
                f"  method=hard\n"
                f"  overlap_lr=0\n"
                f"  overlap_hr=0\n"
                f"  tile_size_lr={model_lr_tile}\n"
                f"  tile_size_hr={contract_hr_tile}"
            )
            log.info(
                f"running hard tiling over {len(nonoverlap_y)}x{len(nonoverlap_x)} grid\n"
                f"  overlap_lr=0\n"
                f"  overlap_hr=0"
            )
            with rasterio.open(pred_path, "w", **pred_profile) as pred_ds:
                for _, _, y0, x0 in iter_window_origins(
                    nonoverlap_y,
                    nonoverlap_x,
                    show_progress=show_progress,
                    desc="non-overlap pass",
                ):
                    lr_y0 = y0 // model_scale
                    lr_x0 = x0 // model_scale
                    depth_tile = replace_nodata_with_zero(
                        depth_ds.read(
                            1,
                            window=Window(lr_x0, lr_y0, model_lr_tile, model_lr_tile),
                            boundless=True,
                            fill_value=0.0,
                        ).astype(np.float32, copy=False),
                        depth_ds.nodata,
                    )
                    dem_tile = replace_nodata_with_zero(
                        dem_ds.read(
                            1,
                            window=Window(x0, y0, contract_hr_tile, contract_hr_tile),
                            boundless=True,
                            fill_value=0.0,
                        ).astype(np.float32, copy=False),
                        dem_ds.nodata,
                    )
                    assert depth_tile.shape == (model_lr_tile, model_lr_tile), (
                        f"depth tile shape {depth_tile.shape} != {(model_lr_tile, model_lr_tile)}"
                    )
                    assert dem_tile.shape == (contract_hr_tile, contract_hr_tile), (
                        f"DEM tile shape {dem_tile.shape} != {(contract_hr_tile, contract_hr_tile)}"
                    )
                    pred_depth_m = self._predict_tile_depth_m(
                        depth_tile=depth_tile,
                        dem_tile=dem_tile,
                        max_depth=max_depth,
                        dem_pct_clip=dem_pct_clip,
                        tile_dem_stats_l=tile_dem_stats_l,
                    )
                    actual_h = max(min(contract_hr_tile, crop_h - y0), 0)
                    actual_w = max(min(contract_hr_tile, crop_w - x0), 0)
                    if actual_h == 0 or actual_w == 0:
                        continue
                    pred_ds.write(
                        pred_depth_m[:actual_h, :actual_w].astype(np.float32, copy=False),
                        1,
                        window=Window(x0, y0, actual_w, actual_h),
                    )
        return len(tile_dem_stats_l), self._summarize_tile_dem_stats(tile_dem_stats_l)

    def run(
        self,
        depth_lr_fp: str | Path,
        dem_hr_fp: str | Path,
        output_fp: str | Path,
        crs_policy: str = "strict",
        max_depth: float | None = None,
        dem_pct_clip: float | None = None,
        window_method: str = "feather",
        tile_overlap: int | None = None,
        tile_size: int | None = None,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Run model-specific ToHR from platform-preprocessed input rasters."""
        start = time.perf_counter()
        log = self.log
        assert self.engine is not None, "worker must be used under context management"

        # Resolve and validate preprocessed input/output paths.
        depth_lr_path = Path(depth_lr_fp).expanduser().resolve()
        dem_hr_path = Path(dem_hr_fp).expanduser().resolve()
        out_path = Path(output_fp).expanduser().resolve()
        assert depth_lr_path.exists(), f"preprocessed low-res depth raster does not exist: {depth_lr_path}"
        assert dem_hr_path.exists(), f"preprocessed DEM raster does not exist: {dem_hr_path}"
        window_method = (window_method or "feather").strip().lower()
        assert window_method in {"hard", "feather"}, f"unsupported window_method={window_method}"
        assert isinstance(show_progress, bool), f"show_progress must be bool, got {type(show_progress)!r}"

        log.info(
            f"starting tohr inference with model_version={self.model_version}\n"
            f"model\n    {self.model_fp}\n"
            f"platform_depth_lr\n    {depth_lr_path}\n"
            f"platform_dem_hr\n    {dem_hr_path}\n"
            f"output\n    {out_path}"
        )

        with rasterio.open(depth_lr_path) as depth_meta_ds, rasterio.open(dem_hr_path) as dem_meta_ds:
            depth_lr_profile = depth_meta_ds.profile.copy()
            dem_platform_profile = dem_meta_ds.profile.copy()
            depth_lr_shape = (int(depth_meta_ds.height), int(depth_meta_ds.width))
            dem_raw_shape = (int(dem_meta_ds.height), int(dem_meta_ds.width))
            dem_platform_nodata = dem_meta_ds.nodata
        depth_crs = depth_lr_profile.get("crs")
        dem_crs = dem_platform_profile.get("crs")
        assert depth_crs is not None and dem_crs is not None, "platform-preprocessed rasters must define CRS"
        assert depth_crs == dem_crs, f"platform-preprocessed CRS mismatch: depth={depth_crs}, dem={dem_crs}"
        depth_bounds = _profile_bounds(depth_lr_profile)
        dem_bounds = _profile_bounds(dem_platform_profile)
        assert all(np.isclose(a, b, atol=1e-6, rtol=0.0) for a, b in zip(depth_bounds, dem_bounds)), (
            f"platform-preprocessed bounds mismatch: depth={depth_bounds}, dem={dem_bounds}"
        )
        log.info(
            "platform-preprocessed inputs\n"
            f"  depth_lr shape={depth_lr_shape} res={_pixel_size_m(depth_lr_profile)} m/pix\n"
            f"  dem_hr shape={dem_raw_shape} res={_pixel_size_m(dem_platform_profile)} m/pix"
        )

        # Resolve model-specific preprocessing settings and runtime contract.
        preprocess_cfg = resolve_preprocess_config(
            self.model_fp,
            max_depth=max_depth,
            dem_pct_clip=dem_pct_clip,
            logger=log,
        )
        assert self.engine.contract is not None, "engine contract must be available"
        contract_scale = int(self.engine.contract.scale)
        contract_lr_tile = int(self.engine.contract.depth_lr_hwc[0])
        contract_hr_tile = int(self.engine.contract.dem_hr_hwc[0])

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

        target_hr_h = int(depth_lr_shape[0] * model_scale)
        target_hr_w = int(depth_lr_shape[1] * model_scale)
        assert target_hr_h > 0 and target_hr_w > 0, f"target HR shape invalid {(target_hr_h, target_hr_w)}"
        dem_model_transform = bounds_to_transform(*depth_bounds, width=target_hr_w, height=target_hr_h)
        dem_model_profile = dem_platform_profile.copy()
        dem_model_profile.update(
            {
                "height": int(target_hr_h),
                "width": int(target_hr_w),
                "transform": dem_model_transform,
            }
        )
        dem_raw_profile = dem_platform_profile.copy()
        dem_raw_profile.update(
            {
                "height": int(dem_raw_shape[0]),
                "width": int(dem_raw_shape[1]),
                "transform": dem_platform_profile["transform"],
            }
        )
        was_resampled = bool(
            (target_hr_h, target_hr_w) != dem_raw_shape
            or not all(
                np.isclose(
                    (dem_model_transform.a, dem_model_transform.e),
                    (dem_platform_profile["transform"].a, dem_platform_profile["transform"].e),
                )
            )
        )
        prepped = {
            "depth_lr_prepared_fp": str(depth_lr_path),
            "dem_hr_prepared_fp": str(dem_hr_path),
            "depth_lr_profile": depth_lr_profile,
            "dem_profile": dem_model_profile,
            "dem_raw_profile": dem_raw_profile,
            "depth_lr_shape": tuple(depth_lr_shape),
            "dem_hr_shape": (int(target_hr_h), int(target_hr_w)),
            "dem_raw_shape": tuple(dem_raw_shape),
            "resampled": was_resampled,
            "crs_policy": crs_policy,
        }
        expected_bounds = _profile_bounds(prepped["depth_lr_profile"])
        log.info("model preprocessing complete")
        log.debug(
            "model preprocessing complete\n"
            f"  scale={model_scale} (HR/LR ratio)\n"
            f"  crs_policy={prepped['crs_policy']}\n"
            f"  aligned depth shape={prepped['depth_lr_shape']} resampled={prepped['resampled']}\n"
            f"  aligned dem shape={prepped['dem_hr_shape']} raw_dem_shape={prepped['dem_raw_shape']}\n"
            f"  max_depth={float(preprocess_cfg['max_depth'])} dem_pct_clip={float(preprocess_cfg['dem_pct_clip'])}"
        )
        execution_path = self._resolve_execution_path(window_method, prepped["dem_raw_shape"])
        log.info(
            "tohr execution path\n"
            f"  window_method={window_method}\n"
            f"  execution_path={execution_path}\n"
            f"  model_space_shape={prepped['dem_hr_shape']}\n"
            f"  raw_output_shape={prepped['dem_raw_shape']}"
        )
        tile_cache_size = 0
        tile_dem_stats = None
        post_resampled = tuple(prepped["dem_raw_shape"]) != tuple(prepped["dem_hr_shape"])
        if execution_path == "simple":
            depth_lr_raw, _, _ = _read_single_band_raster(depth_lr_path)
            dem_platform_raw, _, _ = _read_single_band_raster(dem_hr_path)
            dem_model = np.empty((target_hr_h, target_hr_w), dtype=np.float32)
            reproject(
                source=dem_platform_raw,
                destination=dem_model,
                src_transform=dem_platform_profile["transform"],
                src_crs=dem_platform_profile["crs"],
                src_nodata=dem_platform_nodata,
                dst_transform=dem_model_transform,
                dst_crs=depth_lr_profile["crs"],
                dst_nodata=dem_platform_nodata,
                resampling=Resampling.bilinear,
                num_threads=1,
            )
            dem_model = replace_nodata_with_zero(dem_model, dem_platform_nodata)
            assert np.isfinite(dem_model).all(), "model-space DEM contains non-finite values"
            prediction_model_m, tile_cache_size, tile_dem_stats = self._run_tiled_model_on_prepared(
                depth_lr_raw=depth_lr_raw,
                dem_hr_raw=dem_model,
                depth_lr_profile=depth_lr_profile,
                dem_hr_profile=dem_model_profile,
                preprocess_cfg=preprocess_cfg,
                model_lr_tile=model_lr_tile,
                model_scale=model_scale,
                contract_hr_tile=contract_hr_tile,
                window_method=window_method,
                overlap_lr=overlap_lr,
                show_progress=show_progress,
            )
            assert prediction_model_m.shape == tuple(prepped["dem_hr_shape"]), (
                f"prediction shape {prediction_model_m.shape} must match preprocessed DEM shape {prepped['dem_hr_shape']}"
            )
            output_profile = prepped["dem_raw_profile"].copy()
            output_profile.update(dtype="float32", count=1)
            output_profile.pop("blockxsize", None)
            output_profile.pop("blockysize", None)
            prediction_out_m = prediction_model_m.astype(np.float32, copy=False)
            if post_resampled:
                log.info(
                    f"post-resampling model output from {prediction_model_m.shape} "
                    f"to {tuple(prepped['dem_raw_shape'])} on raw DEM grid with bilinear interpolation."
                )
                prediction_resampled_m = np.empty(tuple(prepped["dem_raw_shape"]), dtype=np.float32)
                if show_progress:
                    with tqdm(total=1, desc="post-resample pass", unit="step") as pbar:
                        reproject(
                            source=prediction_model_m.astype(np.float32, copy=False),
                            destination=prediction_resampled_m,
                            src_transform=prepped["dem_profile"]["transform"],
                            src_crs=prepped["dem_profile"]["crs"],
                            dst_transform=prepped["dem_raw_profile"]["transform"],
                            dst_crs=prepped["dem_raw_profile"]["crs"],
                            resampling=Resampling.bilinear,
                            num_threads=1,
                        )
                        pbar.update(1)
                else:
                    reproject(
                        source=prediction_model_m.astype(np.float32, copy=False),
                        destination=prediction_resampled_m,
                        src_transform=prepped["dem_profile"]["transform"],
                        src_crs=prepped["dem_profile"]["crs"],
                        dst_transform=prepped["dem_raw_profile"]["transform"],
                        dst_crs=prepped["dem_raw_profile"]["crs"],
                        resampling=Resampling.bilinear,
                        num_threads=1,
                    )
                prediction_out_m = prediction_resampled_m
            out_written_fp = self._write_prediction_array(
                output_fp=out_path,
                prediction_out_m=prediction_out_m,
                output_profile=output_profile,
                max_depth=float(preprocess_cfg["max_depth"]),
                show_progress=show_progress,
            )
        else:
            assert window_method == "hard", "windowed path only supports hard windows"
            with tempfile.TemporaryDirectory(prefix="floodsr-windowed-model-") as tmp_dir:
                output_tile_dir = out_path.parent / f"{out_path.stem}__tohr_tiles"
                if output_tile_dir.exists():
                    shutil.rmtree(output_tile_dir)
                output_tile_dir.mkdir(parents=True, exist_ok=True)
                output_tile_fp = output_tile_dir / "tile_r0000000_c0000000.tif"
                dem_model_fp = Path(tmp_dir) / f"{dem_hr_path.stem}_model_dem.tif"
                prediction_model_fp = Path(tmp_dir) / f"{out_path.stem}_model_pred.tif"
                dem_model_profile = _build_single_band_profile(
                    dem_model_fp,
                    dem_platform_profile,
                    target_hr_h,
                    target_hr_w,
                    dem_model_transform,
                )
                with rasterio.open(dem_hr_path) as src_ds, rasterio.open(dem_model_fp, "w", **dem_model_profile) as dst_ds:
                    reproject(
                        source=rasterio.band(src_ds, 1),
                        destination=rasterio.band(dst_ds, 1),
                        src_transform=src_ds.transform,
                        src_crs=src_ds.crs,
                        src_nodata=dem_platform_nodata,
                        dst_transform=dem_model_transform,
                        dst_crs=depth_lr_profile["crs"],
                        dst_nodata=dem_platform_nodata,
                        resampling=Resampling.bilinear,
                        num_threads=1,
                    )
                _zero_nodata_in_place(dem_model_fp, dem_platform_nodata)
                tile_cache_size, tile_dem_stats = self._run_tiled_model_on_prepared_windowed_hard(
                    depth_lr_fp=depth_lr_path,
                    dem_hr_fp=dem_model_fp,
                    prediction_model_fp=prediction_model_fp,
                    preprocess_cfg=preprocess_cfg,
                    model_lr_tile=model_lr_tile,
                    model_scale=model_scale,
                    contract_hr_tile=contract_hr_tile,
                    show_progress=show_progress,
                )
                output_profile = _build_single_band_profile(
                    output_tile_fp,
                    prepped["dem_raw_profile"],
                    prepped["dem_raw_shape"][0],
                    prepped["dem_raw_shape"][1],
                    prepped["dem_raw_profile"]["transform"],
                )
                if post_resampled:
                    log.info(
                        f"post-resampling model output from {tuple(prepped['dem_hr_shape'])} "
                        f"to {tuple(prepped['dem_raw_shape'])} on raw DEM grid with bilinear interpolation."
                    )
                with rasterio.open(prediction_model_fp) as src_ds, rasterio.open(output_tile_fp, "w", **output_profile) as dst_ds:
                    reproject(
                        source=rasterio.band(src_ds, 1),
                        destination=rasterio.band(dst_ds, 1),
                        src_transform=src_ds.transform,
                        src_crs=src_ds.crs,
                        dst_transform=output_profile["transform"],
                        dst_crs=output_profile["crs"],
                        resampling=Resampling.bilinear,
                        num_threads=1,
                    )
                self._postprocess_output_in_place(output_tile_fp, float(preprocess_cfg["max_depth"]))
                out_written_fp = (
                    self._build_windowed_output_vrt(out_path, [output_tile_fp], output_profile.get("nodata"))
                    if self._gdal_is_available()
                    else output_tile_fp
                )

        prepared_dem_bounds = _profile_bounds(prepped["dem_raw_profile"])
        assert all(np.isclose(a, b, atol=1e-6, rtol=0.0) for a, b in zip(prepared_dem_bounds, expected_bounds)), (
            f"output profile bounds {prepared_dem_bounds} do not match expected low-res bounds {expected_bounds}"
        )
        with rasterio.open(out_written_fp) as written_ds:
            written_profile = written_ds.profile.copy()
        written_shape = (int(written_profile["height"]), int(written_profile["width"]))
        assert written_shape == tuple(prepped["dem_raw_shape"]), (
            f"written output shape {written_shape} must match raw DEM shape {prepped['dem_raw_shape']}"
        )
        written_bounds = _profile_bounds(written_profile)
        assert all(np.isclose(a, b, atol=1e-6, rtol=0.0) for a, b in zip(written_bounds, expected_bounds)), (
            f"written output bounds {written_bounds} must match expected low-res bounds {expected_bounds}"
        )

        runtime_s = time.perf_counter() - start
        out_file_size = int(out_written_fp.stat().st_size)
        log.info(
            f"finished tohr inference in {runtime_s:.3f}s; wrote {out_file_size:,} bytes to\n    {out_written_fp}"
        )
        return {
            "output_fp": str(out_written_fp),
            "runtime_s": float(runtime_s),
            "model_version": self.model_version,
            "model_fp": str(self.model_fp),
            "output_size_bytes": out_file_size,
            "execution_path": execution_path,
            "preprocess": {
                "max_depth": float(preprocess_cfg["max_depth"]),
                "dem_pct_clip": float(preprocess_cfg["dem_pct_clip"]),
                "dem_ref_stats": preprocess_cfg["dem_ref_stats"],
                "window_method": window_method,
                "crs_policy": prepped["crs_policy"],
                "tile_overlap_lr": overlap_lr,
                "tile_size_lr": model_lr_tile,
                "tile_size_hr": contract_hr_tile,
                "model_scale": model_scale,
                "tile_cache_size": tile_cache_size,
                "tile_dem_stats": tile_dem_stats,
                "input_shape": {
                    "crop_height": int(prepped["dem_raw_shape"][0]),
                    "crop_width": int(prepped["dem_raw_shape"][1]),
                    "model_space_crop_height": int(prepped["dem_hr_shape"][0]),
                    "model_space_crop_width": int(prepped["dem_hr_shape"][1]),
                    "aligned_depth_shape": [int(x) for x in prepped["depth_lr_shape"]],
                    "aligned_dem_shape": [int(x) for x in prepped["dem_hr_shape"]],
                    "output_shape": [int(x) for x in prepped["dem_raw_shape"]],
                },
                "prepared_inputs": {
                    "depth_lr_prepared_fp": str(prepped["depth_lr_prepared_fp"]),
                    "dem_hr_prepared_fp": str(prepped["dem_hr_prepared_fp"]),
                    "prepped_depth_was_resampled": bool(prepped["resampled"]),
                    "prepped_dem_was_resampled": bool(prepped["resampled"]),
                    "post_sr_was_resampled": bool(post_resampled),
                    "execution_path": execution_path,
                },
            },
        }
