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
- Tiling method is not user-configurable from CLI.
  - users can configure tiling parameters (e.g., overlap, tile size where valid for the model contract)
  - users cannot choose algorithm family (`hard` vs `feather`) at runtime
- Default implementation uses sliding windows with overlap and weighted blending.
- Read windows on demand; do not materialize full input raster in memory.

## Consequences

- All model workers share identical tiling/mosaicking behavior by default.
- Future model workers can reuse common tiling code while keeping model-specific `run()` implementations focused on model logic.
- CLI surface is simpler because tiling strategy is fixed, while still exposing useful numeric parameters.
