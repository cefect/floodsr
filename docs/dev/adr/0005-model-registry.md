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
- Keep model-specific contracts in `docs/dev/adr/models/` rather than embedding them in this shared ADR.

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
  - a matching model worker module exists in `floodsr/models/`, or
  - the version is supported as a built-in worker per its model ADR under `docs/dev/adr/models/`.

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
  - examples:
    - `floodsr/models/ResUNet_16x_DEM.py`
    - `floodsr/models/CostGrow_Terrain.py`
- Subclass behavior:
  - override `run(...)`
  - organize model-specific flow into submethods:
    - Input boundary assertions (validate platform-model boundary artifacts from `0009-preproccessing.md`)
    - Model-specific pre-processing (for example model-space resampling, model-value normalization/log scaling)
    - Tiling/windowing
    - Core inference at model-engine boundary
    - Mosaicking/stitching
    - Model specific post-processing
  - own the model-phase tiling implementation, but call shared tiling/windowing/mosaicking helpers from `tiling.py` (do not duplicate shared tiling primitives inside workers)
  - implement the shared mosaicking vocabulary from `ADR-0008` rather than model-specific method names
  - workers may support both:
    - a simple in-memory array path
    - a raster-backed windowed path for large scenes
  - whole-scene algorithms must still provide a model-phase large-raster tiling strategy; upstream preprocessing alone is not sufficient as the only OOM guard

## ToHR Lifecycle Contract

- `tohr` pipeline resolves model version/artifact and instantiates the matching model worker.
- `tohr` executes the worker under context management:
  - `with model_worker as worker:`
  - `worker.run(...)`
- `tohr` pipeline is responsible for teardown and returning final diagnostics/output metadata.

## Model-Specific ADRs

- Model-specific contracts and implementation notes live in `docs/dev/adr/models/`.
- Current model ADRs:
  - `docs/dev/adr/models/0001-resunet-16x-dem.md`
  - `docs/dev/adr/models/0002-costgrow-terrain.md`
- This ADR owns only the shared registry/worker contract used by all models.
