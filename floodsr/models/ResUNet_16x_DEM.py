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

import logging, math, tempfile, time
from pathlib import Path
from typing import Any

import numpy as np

from floodsr.engine import EngineORT
from floodsr.models.base import Model
from floodsr.preprocessing import _read_single_band_raster, _write_single_band_raster, resolve_preprocess_config, write_prepared_rasters
from floodsr.tiling import build_feather_ramp, build_tile_starts, iter_window_origins


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

    def __init__(
        self,
        model_fp: str | Path,
        *,
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

    def _run_tiled_model_on_prepared(
        self,
        *,
        depth_lr_fp: str | Path,
        dem_hr_fp: str | Path,
        preprocess_cfg: dict[str, object],
        model_lr_tile: int,
        model_scale: int,
        contract_hr_tile: int,
        window_method: str,
        overlap_lr: int,
    ) -> tuple[np.ndarray, int, dict[str, float] | None]:
        """
        Run tiled model execution over prepared rasters and return model-space SR in meter domain.

        Parameters
        ----------
        depth_lr_fp:
            Path to prepared low-resolution depth raster.
        dem_hr_fp:
            Path to prepared high-resolution DEM raster.
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

        Returns
        -------
        tuple[np.ndarray, int, dict[str, float] | None]
            Model-space SR depth in meters, tile-cache size, and summary of tile DEM stats.
        """
        log = self.log
        assert self.engine is not None, "worker must be entered before running inference"
        assert window_method in {"hard", "feather"}, f"unsupported window_method={window_method}"

        # Read already-aligned rasters prepared for model-space execution.
        depth_lr_raw, _depth_lr_nodata, depth_lr_profile = _read_single_band_raster(depth_lr_fp)
        dem_hr_raw, _dem_hr_nodata, dem_hr_profile = _read_single_band_raster(dem_hr_fp)
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
        def _predict_tile_depth_m(y0: int, x0: int) -> np.ndarray:
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
            assert pred_depth_m.shape == (contract_hr_tile, contract_hr_tile), (
                f"prediction shape {pred_depth_m.shape} != {(contract_hr_tile, contract_hr_tile)}"
            )
            dem_stats_used = run_result.get("dem_stats_used")
            if isinstance(dem_stats_used, dict):
                tile_dem_stats_l.append(
                    {
                        "dem_p_clip": float(dem_stats_used.get("p_clip", 0.0)),
                        "dem_min": float(dem_stats_used.get("dem_min", 0.0)),
                        "dem_max": float(dem_stats_used.get("dem_max", 0.0)),
                    }
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
                use_progress=True,
                desc="non-overlap pass",
            ):
                pred_depth_m = _predict_tile_depth_m(y0, x0)
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
                use_progress=True,
                desc="feather pass",
            ):
                pred_depth_m = _predict_tile_depth_m(y0, x0)
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

        tile_dem_stats_summary = None
        if tile_dem_stats_l:
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
            tile_dem_stats_summary = {
                "tile_count": float(dem_stats_np.shape[0]),
                "dem_p_clip_min": float(dem_stats_np[:, 0].min()),
                "dem_p_clip_mean": float(dem_stats_np[:, 0].mean()),
                "dem_p_clip_max": float(dem_stats_np[:, 0].max()),
                "dem_range_min": float(dem_range_np.min()),
                "dem_range_mean": float(dem_range_np.mean()),
                "dem_range_max": float(dem_range_np.max()),
            }

        prediction_depth_m = np.clip(sr_pad[:crop_h, :crop_w], 0.0, max_depth).astype(np.float32, copy=False)
        assert prediction_depth_m.ndim == 2, f"prediction must be 2D; got {prediction_depth_m.shape}"
        return prediction_depth_m, len(tile_cache), tile_dem_stats_summary

    def run(
        self,
        *,
        depth_lr_fp: str | Path,
        dem_hr_fp: str | Path,
        output_fp: str | Path,
        max_depth: float | None = None,
        dem_pct_clip: float | None = None,
        window_method: str = "feather",
        tile_overlap: int | None = None,
        tile_size: int | None = None,
    ) -> dict[str, Any]:
        """
        Run the model-specific ToHR workflow.

        Parameters
        ----------
        depth_lr_fp:
            Low-resolution depth raster path (single band).
        dem_hr_fp:
            High-resolution DEM raster path (single band).
        output_fp:
            Output raster path for high-resolution depth prediction.
        max_depth:
            Optional max-depth override for depth scaling.
        dem_pct_clip:
            Optional DEM percentile clip override.
        window_method:
            Mosaicing mode (`hard` or `feather`).
        tile_overlap:
            Optional LR overlap for feather mode.
        tile_size:
            Optional LR tile-size override (must match model contract).

        Returns
        -------
        dict[str, Any]
            Output path plus runtime and preprocessing diagnostics.
        """
        start = time.perf_counter()
        log = self.log
        assert self.engine is not None, "worker must be used under context management"

        # Resolve and validate user-facing paths.
        depth_lr_path = Path(depth_lr_fp).expanduser().resolve()
        dem_hr_path = Path(dem_hr_fp).expanduser().resolve()
        out_path = Path(output_fp).expanduser().resolve()
        assert depth_lr_path.exists(), f"low-res depth raster does not exist: {depth_lr_path}"
        assert dem_hr_path.exists(), f"DEM raster does not exist: {dem_hr_path}"
        window_method = (window_method or "feather").strip().lower()
        assert window_method in {"hard", "feather"}, f"unsupported window_method={window_method}"

        log.info(
            f"starting tohr inference with model_version={self.model_version}\n"
            f"model\n    {self.model_fp}\n"
            f"depth_lr\n    {depth_lr_path}\n"
            f"dem_hr\n    {dem_hr_path}\n"
            f"output\n    {out_path}"
        )

        # Read raw metadata now so we can assert output georeferencing later.
        depth_lr_raw, _, depth_lr_raw_profile = _read_single_band_raster(depth_lr_path)
        dem_hr_raw, _, dem_hr_raw_profile = _read_single_band_raster(dem_hr_path)
        depth_lr_bounds = _profile_bounds(depth_lr_raw_profile)
        log.info(
            "raw inputs\n"
            f"  depth_lr shape={depth_lr_raw.shape} res={_pixel_size_m(depth_lr_raw_profile)} m/pix\n"
            f"  dem_hr shape={dem_hr_raw.shape} res={_pixel_size_m(dem_hr_raw_profile)} m/pix"
        )

        # Resolve preprocessing config and runtime contract for this model.
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

        # Execute preprocess -> tiled model run -> postprocess write.
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
                f"  aligned dem shape={prepped['dem_hr_shape']} "
                f"raw_dem_shape={prepped['dem_raw_shape']}\n"
                f"  max_depth={float(preprocess_cfg['max_depth'])} "
                f"dem_pct_clip={float(preprocess_cfg['dem_pct_clip'])}"
            )

            prediction_model_m, tile_cache_size, tile_dem_stats = self._run_tiled_model_on_prepared(
                depth_lr_fp=prepped["depth_lr_prepared_fp"],
                dem_hr_fp=prepped["dem_hr_prepared_fp"],
                preprocess_cfg=preprocess_cfg,
                model_lr_tile=model_lr_tile,
                model_scale=model_scale,
                contract_hr_tile=contract_hr_tile,
                window_method=window_method,
                overlap_lr=overlap_lr,
            )
            assert prediction_model_m.shape == tuple(prepped["dem_hr_shape"]), (
                f"prediction shape {prediction_model_m.shape} must match preprocessed DEM shape {prepped['dem_hr_shape']}"
            )

            output_profile = prepped["dem_raw_profile"].copy()
            output_profile.update(dtype="float32", count=1)
            output_profile.pop("blockxsize", None)
            output_profile.pop("blockysize", None)

            prediction_out_m = prediction_model_m.astype(np.float32, copy=False)
            post_resampled = tuple(prepped["dem_raw_shape"]) != tuple(prediction_model_m.shape)
            if post_resampled:
                from rasterio.warp import Resampling, reproject

                log.info(
                    f"post-resampling model output from {prediction_model_m.shape} "
                    f"to {tuple(prepped['dem_raw_shape'])} on raw DEM grid with bilinear interpolation."
                )
                prediction_resampled_m = np.empty(tuple(prepped["dem_raw_shape"]), dtype=np.float32)
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

            prediction_out_m = np.clip(prediction_out_m, 0.0, float(preprocess_cfg["max_depth"])).astype(
                np.float32,
                copy=False,
            )
            prediction_out_m = np.where(
                prediction_out_m < float(self.low_depth_mask_m),
                0.0,
                prediction_out_m,
            ).astype(np.float32, copy=False)

            prepared_dem_bounds = _profile_bounds(prepped["dem_raw_profile"])
            assert all(np.isclose(a, b, atol=1e-6, rtol=0.0) for a, b in zip(prepared_dem_bounds, depth_lr_bounds)), (
                f"output profile bounds {prepared_dem_bounds} do not match incoming low-res bounds {depth_lr_bounds}"
            )

            out_written_fp = _write_single_band_raster(out_path, prediction_out_m, output_profile)
            _, _, written_profile = _read_single_band_raster(out_written_fp)
            written_shape = (int(written_profile["height"]), int(written_profile["width"]))
            assert written_shape == tuple(prepped["dem_raw_shape"]), (
                f"written output shape {written_shape} must match raw DEM shape {prepped['dem_raw_shape']}"
            )
            written_bounds = _profile_bounds(written_profile)
            assert all(np.isclose(a, b, atol=1e-6, rtol=0.0) for a, b in zip(written_bounds, depth_lr_bounds)), (
                f"written output bounds {written_bounds} must match incoming low-res bounds {depth_lr_bounds}"
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
            "preprocess": {
                "max_depth": float(preprocess_cfg["max_depth"]),
                "dem_pct_clip": float(preprocess_cfg["dem_pct_clip"]),
                "dem_ref_stats": preprocess_cfg["dem_ref_stats"],
                "window_method": window_method,
                "tile_overlap_lr": overlap_lr,
                "tile_size_lr": model_lr_tile,
                "tile_size_hr": contract_hr_tile,
                "model_scale": model_scale,
                "tile_cache_size": tile_cache_size,
                "tile_dem_stats": tile_dem_stats,
                "input_shape": {
                    "crop_height": int(prediction_out_m.shape[0]),
                    "crop_width": int(prediction_out_m.shape[1]),
                    "model_space_crop_height": int(prediction_model_m.shape[0]),
                    "model_space_crop_width": int(prediction_model_m.shape[1]),
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
                },
            },
        }
