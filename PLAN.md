

## Scope


What is already done:
- `dev_cg` already contains the clean CostGrow migration and passing tests.
- `ResUNet_16x_DEM` already appears to satisfy the current model-phase tiling baseline from [`docs/dev/adr/models/0001-resunet-16x-dem.md`](/workspace/docs/dev/adr/models/0001-resunet-16x-dem.md):
  - `simple + hard`
  - `simple + feather`
  - `windowed + hard`
- platform preprocessing and HRDEM fetch already have windowed large-raster paths.


What remains:
- `CostGrow_Terrain` still relies on shared preprocessing/materialization for large rasters and does not yet own a true model-phase large-raster strategy.
- this is the main ADR gap called out by:
  - [`docs/dev/adr/0008-tiling.md`](/workspace/docs/dev/adr/0008-tiling.md)
  - [`docs/dev/adr/0005-model-registry.md`](/workspace/docs/dev/adr/0005-model-registry.md)
  - [`docs/dev/adr/models/0002-costgrow-terrain.md`](/workspace/docs/dev/adr/models/0002-costgrow-terrain.md)


## Cross References


- [`docs/dev/adr/0005-model-registry.md`](/workspace/docs/dev/adr/0005-model-registry.md)
- [`docs/dev/adr/0008-tiling.md`](/workspace/docs/dev/adr/0008-tiling.md)
- [`docs/dev/adr/0009-preproccessing.md`](/workspace/docs/dev/adr/0009-preproccessing.md)
- [`docs/dev/adr/0010-DEM-fetch.md`](/workspace/docs/dev/adr/0010-DEM-fetch.md)
- [`docs/dev/adr/models/0001-resunet-16x-dem.md`](/workspace/docs/dev/adr/models/0001-resunet-16x-dem.md)
- [`docs/dev/adr/models/0002-costgrow-terrain.md`](/workspace/docs/dev/adr/models/0002-costgrow-terrain.md)
- [`floodsr/models/ResUNet_16x_DEM.py`](/workspace/floodsr/models/ResUNet_16x_DEM.py)
- [`floodsr/models/CostGrow_Terrain.py`](/workspace/floodsr/models/CostGrow_Terrain.py)
- [`floodsr/tiling.py`](/workspace/floodsr/tiling.py)
- [`floodsr/tohr.py`](/workspace/floodsr/tohr.py)


## Compliance Read


`ResUNet_16x_DEM`:
- appears compliant with the current ADR baseline
- already has model-phase `windowed + hard`
- already has `simple + feather`
- does not need major new feature work on this branch unless we find a shared-helper extraction that clearly reduces duplication without changing behavior


`CostGrow_Terrain`:
- not yet compliant with the ADR requirement that whole-scene algorithms own a model-phase large-raster strategy
- currently reports `execution_path = "global"` even when platform materialization is `windowed`; here `execution_path` means the model-phase runtime mode recorded in the worker result (`"simple"`/`"windowed"` for tiled model execution, `"global"` for whole-scene execution), so the current value shows that CostGrow still runs as one global solve after preprocessing instead of owning an ADR-compliant model-phase large-raster path. After this feature, there should be no more 'global'.
- still needs a true model-phase `windowed + hard` path
- `windowed + feather` can be saved till later


## Plan


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
- keep the existing CostGrow unit and integration tests
- add one focused large-raster CostGrow test that proves model-phase windowed execution is selected
- add one focused regression or contract test for output artifact expectations if the implementation changes packaging
- no new test module


8. Re-check ADR wording after implementation.
- if the implementation requires narrowing or clarifying the CostGrow ADR, update:
  - [`docs/dev/adr/models/0002-costgrow-terrain.md`](/workspace/docs/dev/adr/models/0002-costgrow-terrain.md)
  - [`docs/dev/adr/0008-tiling.md`](/workspace/docs/dev/adr/0008-tiling.md)
- do not widen ADR promises beyond what is actually implemented on this branch


## Out Of Scope For This Branch


- a broad shared-tiling refactor across every phase
- `ResUNet_16x_DEM` feature expansion beyond maintaining current compliance
- model-phase `windowed + feather` for CostGrow
- tutorials/docs (do this later)



