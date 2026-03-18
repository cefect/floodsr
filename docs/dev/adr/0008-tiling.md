# ADR-0008: Shared Tiling and Mosaicking

## Context

Model workers need the same windowing and mosaicking behavior to avoid duplicated logic and divergent inference outputs across models.

## Decision

- Keep tiling logic in one shared script:
  - `floodsr/tiling.py`
- `tiling.py` contains both:
  - window generation/extraction helpers
  - mosaicking/blending helpers
- Model workers must import and use these shared tiling functions.
- Tiling method may be exposed by the current CLI/API implementation, but shared tiling behavior still lives in `tiling.py`.
- Default implementation uses sliding windows with overlap and weighted blending.
- Keep two execution styles:
  - a simple in-memory path
  - a raster-backed windowed path that reads windows on demand
- In this phase, the raster-backed windowed path only needs to support hard windows. Feathered blending remains on the simple path.

## Consequences

- All model workers share identical tiling/mosaicking behavior by default.
- Future model workers can reuse common tiling code while keeping model-specific `run()` implementations focused on model logic.
- The hard-window raster-backed path gives a memory-bounded option for large scenes without requiring a full rewrite of feathered mosaicking first.
