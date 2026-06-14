# ADR-M-0001: ResUNet_16x_DEM Model Contract

## Context

`ResUNet_16x_DEM` is the default learned super-resolution model shipped by FloodSR.

Shared model architecture, registry, worker lifecycle, tiling ownership, and engine/runtime boundaries are defined in:
- `docs/dev/adr/0005-model-registry.md`
- `docs/dev/adr/0008-tiling.md`
- `docs/dev/adr/0009-preproccessing.md`
- `docs/dev/adr/0015-engine-runtime.md`

This ADR owns the model-specific contract for `ResUNet_16x_DEM`.

## Decision

- `ResUNet_16x_DEM` remains an artifact-backed model version in `floodsr/models.json`.
- Inference artifact format is ONNX (`model_infer.onnx`).
- Related packaged training metadata may include `train_config.json`.
- The worker module is `floodsr/models/ResUNet_16x_DEM.py`.
- Runtime execution uses the shared engine abstraction, with ONNX Runtime owned by `ADR-0015`.

## Model-Engine Boundary Contract

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

## Worker Flow

1. Assert the shared platform-model boundary from `ADR-0009`.
2. Resolve model parameters from packaged metadata such as `train_config.json`.
3. Apply model-specific transforms such as model-space resampling, `log1p` scaling, and tile-local DEM normalization.
4. Execute tiled inference through the shared engine boundary.
5. Stitch tile outputs through the shared tiling contract in `ADR-0008`.
6. Convert predictions back to depth meters, apply low-depth masking, enforce final masking from the prepared DEM valid domain, and materialize the final raster.

## Platform Input Assumptions

- `depth_lr` arrives from the platform boundary as a real-valued depth raster with no surviving mask-only semantics at worker runtime.
- Invalid low-resolution depth cells are expected to have been normalized by preprocessing into the shared prepared-raster contract rather than handled through a separate mask artifact.

## Tiling Notes

- The worker may support both:
  - a simple in-memory path
  - a raster-backed windowed path for large scenes
- The worker owns the model-phase tiling implementation for `ResUNet_16x_DEM`, but it must use shared helpers and shared method vocabulary from `ADR-0008`.
- `ResUNet_16x_DEM` is the current compatibility baseline for model-phase support:
  - `simple + hard`
  - `simple + feather`
  - `windowed + hard`
