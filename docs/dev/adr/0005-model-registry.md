# ADR-0005: Models (Registry, Workers, and I/O Contracts)

## Context

FloodSR needs a stable model layer that:
- discovers available model versions,
- validates/fetches model artifacts,
- maps each model version to model-specific execution code,
- keeps model-specific logic out of CLI and engine modules.

## Decision

- Keep `floodsr/models.json` as the source of available model versions.
- Keep model artifact retrieval/checksum policy in `model_registry.py`.
- Add a project-wide base class `Model` and implement each model as a subclass in its own module.
- Route `tohr` execution through model workers and make the pipeline function responsible for worker creation/teardown.

## Model Registry Contract

- Manifest file: `floodsr/models.json`
- Required fields per model record:
  - `file_name`
  - `url`
  - `sha256`
- Optional fields:
  - `description`
- Cache path pattern:
  - `<cache_dir>/<model_version>/<file_name>`
- A model is runnable only when:
  - the version exists in `models.json`, and
  - a matching model worker module exists in `floodsr/models/`.

## Model Worker Architecture

- Base class:
  - module: `floodsr/models/base.py`
  - name: `Model`
- Shared `Model` methods:
  - logger initialization helper
  - `is_valid(...)`
  - placeholder `run(...)` (must be overridden by subclasses)
  - context manager lifecycle (`__enter__`, `__exit__`) for clean resource management
- Per-model modules:
  - one module per model version
  - naming pattern: `floodsr/models/<normalized_model_version>.py`
  - normalize non-alphanumeric characters in `<model_version>` to `_`
  - example: `floodsr/models/ResUNet_16x_DEM.py`
- Subclass behavior:
  - override `run(...)`
  - organize model-specific flow into submethods:
    - Model specific pre-processing
    - Tiling/windowing
    - Core inference at model-engine boundary
    - Mosaicking/stitching
    - Model specific post-processing
  - call shared tiling/windowing/mosaicking helpers from `tiling.py` (do not duplicate tiling implementations inside workers)
  - tiling method is fixed by implementation (not a user-facing toggle); only tiling parameters are configurable

## ToHR Lifecycle Contract

- `tohr` pipeline resolves model version/artifact and instantiates the matching model worker.
- `tohr` executes the worker under context management:
  - `with model_worker as worker:`
  - `worker.run(...)`
- `tohr` pipeline is responsible for teardown and returning final diagnostics/output metadata.

## Model Types

### 16x DEM-conditioned ResUNet (`ResUNet_16x_DEM`)

#### Artifact

- Inference artifact format: ONNX (`model_infer.onnx`)
- Current release reference: `v2026.02.19`
- Related training metadata (when packaged): `train_config.json`
- Model worker module:
  - `floodsr/models/ResUNet_16x_DEM.py`

#### Model-Engine Boundary Contract

- Tensor names:
  - inputs: `depth_lr`, `dem_hr`
  - output: `depth_hr_pred`
- Tensor layout and dtype:
  - NHWC `float32`, single channel
  - `depth_lr`: `[N, 32, 32, 1]`
  - `dem_hr`: `[N, 512, 512, 1]`
  - `depth_hr_pred`: `[N, 512, 512, 1]`
- Geometry:
  - fixed scale `16` (`512 / 32`)
  - output H/W must match `dem_hr` H/W
- Value-domain:
  - entry tensors finite and normalized to `[0, 1]`
  - output tensor normalized/log-space before inverse transform

#### Workflow (from `others/inference_inline_norm_loop.ipynb`)

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
