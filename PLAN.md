## Scope

What is already done:

- PCRaster runtime gating + install proof.
- Architecture extension: built-in models in registry, `requires_model_artifact` flag, nullable `model_fp`.
- `CostGrow_Terrain`: core algorithm, simple and model-phase `tile_halo` windowed paths, CLI/`tohr` wiring.
- `ResUNet_16x_DEM`: `simple+hard`, `simple+feather`, `windowed+hard`.
- Platform preprocessing and HRDEM fetch have windowed large-raster paths.
- `window_method` drives execution path selection in both models: `windowed` only when `window_method=="hard"` and raster bytes ≥ threshold (same contract as ResUNet).
- Test modules re-organized to match source structure: `test_model_costgrow.py`, `test_model_resunet.py` for model-specific tests; common infra stays in `test_engine_contracts.py`, `test_tohr_regression.py`, etc.

What remains (follow-on work):

- `windowed+feather` for CostGrow is deferred.
- Broader user docs/tutorial cleanup for CostGrow is still a follow-on task.

## Cross References

- [`docs/dev/adr/0005-model-registry.md`](docs/dev/adr/0005-model-registry.md)
- [`docs/dev/adr/0008-tiling.md`](docs/dev/adr/0008-tiling.md)
- [`docs/dev/adr/0009-preproccessing.md`](docs/dev/adr/0009-preproccessing.md)
- [`docs/dev/adr/0010-DEM-fetch.md`](docs/dev/adr/0010-DEM-fetch.md)
- [`docs/dev/adr/models/0001-resunet-16x-dem.md`](docs/dev/adr/models/0001-resunet-16x-dem.md)
- [`docs/dev/adr/models/0002-costgrow-terrain.md`](docs/dev/adr/models/0002-costgrow-terrain.md)
- [`floodsr/models/ResUNet_16x_DEM.py`](floodsr/models/ResUNet_16x_DEM.py)
- [`floodsr/models/CostGrow_Terrain.py`](floodsr/models/CostGrow_Terrain.py)
- [`floodsr/tiling.py`](floodsr/tiling.py)
- [`floodsr/tohr.py`](floodsr/tohr.py)

## Compliance Read

`ResUNet_16x_DEM`:

- appears compliant with the current ADR baseline
- already has model-phase `windowed + hard`
- already has `simple + feather`
- does not need major new feature work on this branch unless we find a shared-helper extraction that clearly reduces duplication without changing behavior

`CostGrow_Terrain`:

- now matches the ADR requirement that whole-scene algorithms own a model-phase large-raster strategy for `windowed + hard`
- uses the same `window_method` execution selector pattern as ResUNet: only `hard` may choose the large-raster `windowed` path
- has an explicit model-owned tile contract: fixed fine-grid core tiles, halo from `dp_coarse_pixel_max`, staged coarse global prefill, and `hard_crop_core` merge on the canonical output grid
- `windowed + feather` can be saved till later

## Plan

## Compliance Matrix

| Phase         | simple+hard | simple+feather | windowed+hard | windowed+feather |
| ------------- | ----------- | -------------- | ------------- | ---------------- |
| DEM fetch     | YES         | NO             | YES           | NO               |
| Preprocessing | YES         | NO             | YES           | NO               |
| ResUNet       | YES         | YES            | YES           | planned          |
| CostGrow      | YES         | YES            | YES (tile_halo) | planned          |

### Completed plan summary

1. Confirm the exact compliance gap in code and tests.
- keep the current passing tests as the baseline
2. Reuse ResUNet only as the behavioral reference, not as a refactor target.
- preserve current `ResUNet_16x_DEM` behavior
- only extract shared tiling helpers from [`floodsr/models/ResUNet_16x_DEM.py`](/workspace/floodsr/models/ResUNet_16x_DEM.py) if the extraction is directly useful for CostGrow and does not create a broad refactor
3. Design the CostGrow model-phase `windowed + hard` strategy.
- keep the small-raster whole-scene path as the `simple` execution mode
- add a distinct model-phase `windowed` execution mode for large rasters
- prefer hard-window writes and avoid feather blending in this phase
- prefer concrete raster output over VRT unless the implementation naturally requires a VRT wrapper
4. Define the CostGrow tile contract.
- decide the halo/context rule for each tile
- decide how coarse-to-fine growth state is localized or staged across tiles
- decide what intermediate state must be written to disk between passes
- keep output support tied to the prepared DEM valid domain
5. Implement the model-phase large-raster path in [`floodsr/models/CostGrow_Terrain.py`](/workspace/floodsr/models/CostGrow_Terrain.py).
- add execution-path selection consistent with the large-raster trigger used by `tohr`
- separate the current whole-scene path from the new tiled path
- return explicit runtime metadata showing whether CostGrow used `simple` or `windowed` model execution
6. Align shared reporting and diagnostics.
- ensure `tohr` result metadata clearly distinguishes:
  - platform materialization
  - model execution path
- keep the reporting language consistent with [`docs/dev/adr/0008-tiling.md`](/workspace/docs/dev/adr/0008-tiling.md)
7. Expand tests only around the real compliance gap.
- keep CostGrow coverage in the model-specific module layout
- add one focused large-raster CostGrow test that proves model-phase windowed execution is selected
- add one focused contract test for the real `tile_halo` execution path and merge behavior
8. Re-check ADR wording after implementation.
- if the implementation requires narrowing or clarifying the CostGrow ADR, update:
  - [`docs/dev/adr/models/0002-costgrow-terrain.md`](/workspace/docs/dev/adr/models/0002-costgrow-terrain.md)
  - [`docs/dev/adr/0008-tiling.md`](/workspace/docs/dev/adr/0008-tiling.md)
- do not widen ADR promises beyond what is actually implemented on this branch
  *CostGrow `windowed+hard` uses a bounded-region `tile_halo` contract: coarse WSE support is staged once, then fine-grid growth/connectivity run per padded tile and only cropped cores are merged.*

## Out Of Scope For This Branch

- `windowed+feather` for either model.
- Broad shared-tiling refactor.
- Tutorials/docs (follow-on PR).
