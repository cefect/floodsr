# ADR-M-0002: CostGrow_Terrain Model Contract

## Context

`CostGrow_Terrain` is a built-in terrain-penalty downscaling method implemented in project code rather than downloaded as a learned weights artifact.

Shared model architecture, registry, worker lifecycle, tiling ownership, and engine/runtime boundaries are defined in:
- `docs/dev/adr/0005-model-registry.md`
- `docs/dev/adr/0008-tiling.md`
- `docs/dev/adr/0009-preproccessing.md`
- `docs/dev/adr/0015-engine-runtime.md`
- `docs/dev/adr/0002-packaging-and-installation-strategy.md`

This ADR owns the model-specific contract for `CostGrow_Terrain`.

## Decision

- `CostGrow_Terrain` is a built-in model version and does not require a downloadable weights artifact.
- The worker module is `floodsr/models/CostGrow_Terrain.py`.
- Runtime execution is worker-owned and does not go through the ONNX engine abstraction.
- `CostGrow_Terrain` requires PCRaster and therefore belongs to the extended install capability, not the basic install path.

## Worker Boundary Contract

- Inputs are the shared platform-preprocessed boundary artifacts from `ADR-0009`.
- The worker produces the same final FloodSR output contract as other models:
  - float32 depth raster
  - non-empty valid prediction domain
  - shared diagnostics/reporting ownership at the `tohr` pipeline boundary
- The worker should treat the prepared low-resolution depth raster as a real-valued depth surface with no separate mask artifact at runtime:
  - dry/source selection is value-driven through `min_depth_threshold`
  - final output support is constrained by the prepared DEM valid domain

## Runtime and Dependency Notes

- PCRaster is treated as an extended-install dependency, not a `pyproject.toml` pip extra. (must install PCRaster with conda). 
- The worker must fail with a clear runtime error when PCRaster is unavailable.
- `floodsr doctor` and model-listing flows must report PCRaster capability without making the base install crash.

## Execution Notes

- `CostGrow_Terrain` consumes the shared preprocessing/materialization path, but it must also own a model-phase large-raster strategy.
- `CostGrow_Terrain` may not rely only on upstream preprocessing/materialization as its sole OOM guard for oversized rasters.
- The worker must implement the shared model-phase mosaicking vocabulary from `ADR-0008`:
  - `hard`
  - `feather`
- Because CostGrow is natively whole-raster, its model-phase tiling implementation may require halos, staged intermediate rasters, or tile-local growth plus a later merge step. Those mechanics are owned by this model ADR, but the window generation and mosaicking primitives should come from `floodsr/tiling.py`.
- The intended end state is a true model-owned large-raster contract:
  - define the tile extent and any halo/context rule
  - define what intermediate state is staged between passes
  - define how growth/connectivity is localized, staged, or merged across tiles
  - define the final output merge rule on the canonical fine grid
- A disk-backed whole-scene implementation is allowed only as an intermediate step. If the expensive CostGrow stages still run across one full-scene domain, that is not yet the final ADR-compliant model-phase tiling design even if fine-grid intermediates are written to memmaps or emitted through block IO.
- Until `windowed + feather` is explicitly implemented and tested for this worker, support for that method should be documented as not yet complete rather than silently treated as equivalent to `hard`.
- Because the CLI contract is depth-based, `min_depth_threshold` is applied to the low-res depth input for CostGrow before coarse WSE reconstruction, so it controls which coarse cells become wet source anchors. The low-resolution depth input should be interpreted as a real-valued depth raster, with dry cells represented by low/zero depth rather than by a separate runtime mask artifact. This is different from the ResUNet path, where `min_depth_threshold` is applied later as a post-inference mask on predicted output depth. The current CostGrow worker default is `1e-3` when the flag is omitted; the notebook has no equivalent threshold because it starts from coarse WSE rather than coarse depth.
- Final CostGrow masking should treat the prepared DEM valid domain as the source of truth for where output may exist.

## differences from POC

- The inline notebook takes coarse WSE plus fine DEM as in-memory `xarray` inputs, while the current `floodsr tohr` CostGrow path takes low-res depth plus high-res DEM filepaths and reconstructs coarse WSE internally as `dem_lr + depth_lr`.
- The core terrain-penalty growth logic is still effectively the same: bilinear coarse-to-fine resampling, wet-above-ground filtering, PCRaster `spreadzone` growth over a terrain-derived cost surface, linear distance decay, and a final connected-component filter anchored on the original wet partials.
- The notebook assumes already-aligned rasters with matching CRS/bounds and an exact integer downscale factor; the current CLI path runs through shared platform preprocessing first, including CRS policy handling and prepared raster materialization before the CostGrow worker starts.
- The notebook derives validity purely from `NaN` masks on the loaded arrays; the current worker reconstructs validity masks from the original source rasters and reprojects those masks onto the prepared coarse and fine grids before selecting wet anchors and writing output.
- The notebook is fully in-memory and `rioxarray`-based; the production worker is `rasterio`/NumPy-based and can use the shared simple or windowed platform-materialization path for large rasters.
- The notebook returns downscaled WSE, and the notebook plot converts that to WSH/depth afterward; the current CLI worker writes the final float32 depth raster directly, with nodata metadata and a structured runtime metadata payload.
- CostGrow is now a built-in model version (`CostGrow_Terrain`) with no weights artifact; the CLI resolves it through the normal `--model-version` path and separately checks PCRaster availability rather than treating it like a downloadable ONNX model.


- The notebook exposes CostGrow-specific knobs directly (`dp_coarse_pixel_max`, `decay_frac`, `distance_fill_method`, `distance_fill_kwargs`); the current CLI only exposes `--min-depth-threshold` for this worker, while the other CostGrow knobs remain internal defaults in `floodsr/models/CostGrow_Terrain.py`. see issue #46. 


 
