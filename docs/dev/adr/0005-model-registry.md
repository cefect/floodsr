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
  - naming pattern: `floodsr/models/<model_version>.py`
  - example: `floodsr/models/4690176_0_1770580046_train_base_16.py`
- Subclass behavior:
  - override `run(...)`
  - organize model-specific flow into submethods for the five typical SR stages:
    - model specific pre-processing
    - tiling/windowing
    - core inference at model-engine boundary
    - mosaicking/stitching
    - model specific post-processing
  - call shared tiling/windowing/mosaicking helpers from `tiling.py` (do not duplicate tiling implementations inside workers)
  - tiling method is fixed by implementation (not a user-facing toggle); only tiling parameters are configurable

## ToHR Lifecycle Contract

- `tohr` pipeline resolves model version/artifact and instantiates the matching model worker.
- `tohr` executes the worker under context management:
  - `with model_worker as worker:`
  - `worker.run(...)`
- `tohr` pipeline is responsible for teardown and returning final diagnostics/output metadata.

## Model Types

### 16x DEM-conditioned ResUNet (`4690176_0_1770580046_train_base_16`)

#### Artifact

- Inference artifact format: ONNX (`model_infer.onnx`)
- Current release reference: `v2026.02.19`
- Related training metadata (when packaged): `train_config.json`
- Model worker module:
  - `floodsr/models/4690176_0_1770580046_train_base_16.py`

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
