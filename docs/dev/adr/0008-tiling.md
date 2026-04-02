# ADR-0008: Shared Tiling and Mosaicking

## Context

FloodSR has three distinct large-raster phases that may each need memory-bounded execution:
- DEM fetch
- platform preprocessing/materialization
- model execution

These phases do not have identical constraints:
- DEM fetch is source-driven and may need source-native tiling rules.
- Preprocessing is target-grid driven and must preserve the platform-model boundary exactly.
- Model execution is algorithm-driven and may require model-specific tile-local work, halos, or iterative passes.

We still need one shared vocabulary and helper layer for:
- window generation
- raster-backed reads/writes
- overlap handling
- mosaicking/blending
- capability reporting

## Decision

- Keep shared tiling/mosaicking helpers in `floodsr/tiling.py`.
- `tiling.py` owns shared primitives, not phase orchestration:
  - window/grid generation helpers
  - source/destination window mapping helpers
  - overlap/halo helpers
  - mosaicking/blending helpers
  - raster-backed block/window iteration helpers
  - common capability metadata/types for phase implementations
- Each large-raster phase owns its own tiling implementation while using the shared helpers in `tiling.py`:
  - DEM fetch tiling is owned by `ADR-0010`
  - preprocessing/materialization tiling is owned by `ADR-0009`
  - model tiling is owned by `ADR-0005` plus each model ADR
- All three phases must implement some large-raster windowed/tiling path. No phase may rely exclusively on full-scene in-memory execution for oversized rasters.
- Keep two execution styles available across phases:
  - `simple`: may materialize the full scene in memory
  - `windowed`: reads and/or writes windows on demand
- For the model-execution phase, `windowed` means more than just disk-backed intermediates:
  - the worker must own a bounded large-raster execution strategy for the model phase itself
  - the expensive model-stage operations should run per tile or per bounded region, not only as one full-scene solve with arrays spilled to disk
  - the worker ADR should document the tile contract, including any halo/context rule, staged intermediate state, and final merge rule
- A disk-backed whole-scene solve may be a transitional implementation, but it is not by itself the final model-phase tiling contract required by this ADR.
- Keep one shared mosaicking-method vocabulary across phases:
  - `hard`: direct writes or last-write/no-weight stitching for non-overlap or cropped windows
  - `feather`: overlap-aware weighted blending
- Feathering is only required for the model-execution phase. (may add more methods later). DEM fetch and preprocessing do not need feather support.
- The user-facing/API-facing tiling method may be exposed by CLI and Python entrypoints, but the implementation for each phase still lives in that phase module and uses shared helpers from `tiling.py`.

### Windowed Output Packaging

- `windowed` defines an execution/materialization strategy, not a required raster packaging format.
- All `windowed` implementations must produce a raster-backed dataset on the canonical phase output grid.
- A `windowed` implementation may return:
  - one concrete raster file, or
  - a VRT that references one or more backing raster files
- Use a VRT when the phase naturally leaves multiple backing rasters or when virtual assembly preserves useful lazy composition.
- Do not require a VRT when the phase already produces one canonical raster file; in that case the concrete raster is the preferred output.

Current implementation summary:
- DEM fetch (HRDEM): `windowed` output is VRT-backed tile assembly.
- preprocessing/materialization: contract is raster-backed prepared rasters on the canonical grid; current implementation writes concrete rasters.
- `ResUNet_16x_DEM`: `windowed + hard` currently writes one concrete raster tile and may expose a VRT wrapper over that raster when GDAL VRT support is available. I guess its nice to fallback to single raster... makes things more complicated through. 
- `CostGrow_Terrain`: current work is still transitional. The branch may use disk-backed intermediates and blockwise raster IO, but the expensive growth/fill stages still operate at whole-scene scope. That reduces memory pressure, but it is not yet the final ADR target for model-phase `windowed` execution.


## Compatibility Matrix

| Phase | Current owner | `simple + hard` | `simple + feather` | `windowed + hard` | `windowed + feather` |
| --- | --- | --- | --- | --- | --- |
| DEM fetch (HRDEM) | `ADR-0010` / `dem_sources/hrdem_mosaic.py` | implemented | no | implemented | no |
| preprocessing | `ADR-0009` / `preprocessing.py` | implemented | no | implemented | no |
| ResUNet_16x_DEM | `ADR-M-0001` / model worker | implemented | implemented | implemented | planned |
| CostGrow_Terrain | `ADR-M-0002` / model worker | n/a | n/a | planned | planned |

Rules:
- This matrix is the shared capability contract for tiling and mosaicking.
- A phase implementation may reject a method only when that limit is documented by its ADR and reflected in this matrix.
- Whole-scene algorithms, including `CostGrow_Terrain`, must still provide a model-phase large-raster tiling strategy for `windowed` execution. Upstream preprocessing alone is not a sufficient OOM guard.
- For avoidance of doubt: a worker does not satisfy this rule merely by moving full-scene arrays to memmaps or block-writing outputs while still running the core model solve globally.


## Consequences

- Fetch, preprocessing, and model execution can evolve independently while staying interoperable.
- The project keeps one shared method vocabulary and one shared helper layer.
- Model workers no longer own the entire tiling architecture; they own only the model-phase implementation built on shared helpers.
- Large-raster support becomes an explicit requirement for every phase, including whole-scene algorithms such as `CostGrow_Terrain`.
- Per `0001-architecture-and-cli.md`, callers must not have explicit output paths silently rewritten to match a mode-specific artifact type; implementations should validate early and fail clearly when a suffix or packaging constraint is required.
