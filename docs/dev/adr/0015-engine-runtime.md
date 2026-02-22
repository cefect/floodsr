# ADR-0015: Inference Engine Runtime Abstraction

## Status

Accepted.

## Context

FloodSR needs a swappable execution backend so model artifacts can run across runtime providers without changing CLI-level workflows.

The codebase must separate:
- model concerns (artifact/version/contract) from
- engine concerns (runtime session/provider/execution).

Model concerns are defined in `ADR-0005`.

## Decision

- Keep a strict engine interface (`EngineBase`) with:
  - `load()`
  - `run_tile(...)`
  - `model_path()`
- MVP runtime implementation is ONNX Runtime via `EngineORT`.
- MVP provider policy:
  - default `providers=["CPUExecutionProvider"]`
- Engine validates runtime session metadata before execution:
  - required input names exist
  - static tensor dimensions align with model contract
  - output geometry is compatible with DEM input geometry

## Engine Responsibilities

- Load model into runtime session.
- Resolve runtime-visible model tensor metadata.
- Build/validate feed dictionaries for runtime execution.
- Execute forward pass and return prediction tensors/results.
- Report runtime/provider diagnostics.
- Release runtime resources when model worker context exits.

## Non-Responsibilities (Handled Outside Engine)

- Model registry, manifest resolution, and checksum validation.
- Model worker lifecycle (`Model` subclass creation, `run()`, and context ownership).
- DEM/depth file I/O and geospatial alignment.
- Global scene tiling and mosaicking strategy.
- Output raster writing and CLI orchestration.

## Consequences

- Runtime backend can evolve independently from model registry and CLI surfaces.
- Additional engines (GPU ORT, TensorRT, PyTorch runtime adapters) can implement `EngineBase` without changing top-level CLI flow.
- The model-engine boundary remains explicit and testable across backends.
