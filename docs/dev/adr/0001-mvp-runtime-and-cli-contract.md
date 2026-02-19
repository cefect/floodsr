# ADR-0001: MVP Runtime and CLI Contract
 

The project needs an MVP inference path that works cross-platform and can be called both from shell scripts and a QGIS plugin subprocess.

## Decision

- Artifact format: ONNX model files.
- Runtime for MVP: ONNX Runtime (CPU).
- User-facing endpoint: CLI.
- Cross-platform target: Windows and Unix.
- GPU support is deferred, but architecture must keep a stable CLI contract while allowing future execution provider swaps (CUDA on Linux, DirectML on Windows, etc.).

## Model I/O Contract (from `dev/infer_test_tiles.ipynb`)

- Model I/O names:
  - inputs: `depth_lr`, `dem_hr`
  - output: `depth_hr_pred`
- Tensor layout and dtype:
  - all tensors are NHWC, `float32`, single channel (`C=1`)
  - `depth_lr`: `[N, 32, 32, 1]`
  - `dem_hr`: `[N, 512, 512, 1]`
  - `depth_hr_pred`: `[N, 512, 512, 1]`
  - `N` is dynamic; MVP notebook uses `N=1`
- Geometry constraints:
  - `depth_lr` and `dem_hr` are square tiles
  - output H/W must match `dem_hr` H/W
  - upscale ratio is `16` (512 / 32)
- Input preprocessing:
  - nodata replacement in notebook POC: nodata pixels are replaced with `0.0` before normalization
  - low-res depth (`depth_lr`) preprocessing:
    - clip to `[0, max_depth]`
    - apply `log1p(depth) / log1p(max_depth)`
    - clamp to `[0, 1]`
  - high-res DEM (`dem_hr`) preprocessing:
    - clip negatives to `0`
    - clip upper bound by DEM clip stat (`p_clip`)
    - min-max normalize with (`dem_min`, `dem_max`)
    - clamp to `[0, 1]`
    - prefer persisted training stats (`dem_stats`) from `train_config.json` when present
- Output postprocessing:
  - `depth_hr_pred` is normalized log-space
  - invert with `expm1(norm * log1p(max_depth))`
  - clamp back to `[0, max_depth]` in depth units
- Runtime/provider policy for MVP:
  - ORT session is created with `providers=["CPUExecutionProvider"]`
  - input names and static dimensions are validated against ORT session metadata before `session.run(...)`

## Consequences

- The inference engine implementation can evolve without breaking CLI callers.
- CPU path is prioritized for reliability and packaging simplicity.
- Future GPU support should be additive, not a rewrite.
