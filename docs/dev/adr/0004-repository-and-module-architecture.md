# ADR-0004: Repository and Module Architecture

 
The codebase should keep the CLI thin, isolate engine/runtime concerns, and preserve a swappable execution backend boundary.

see also:
- ADR-0011: Parameters and Configuration

## Decision

Use a structure centered on these modules:

- CLI surface:
  - `cli.py`
  - `model_registry.py`
  - `checksums.py`
- Cache subsystem:
  - `cache/paths.py`
  - `cache/policy.py`
  - `cache/lifecycle.py`
  - `cache/reporting.py`
- Engine abstraction:
  - `engine/base.py`
  - `engine/ort.py`
  - `engine/providers.py`
- Data model and preprocessing:
  - `schema/config.py`
  - `preprocess/normalize.py`
- Runtime-agnostic tiling/stitching:
  - `tiling/tiles.py`
  - `tiling/stitch.py`
- DEM source abstraction:
  - `dem_sources/base.py`
  - `dem_sources/hrdem_stac.py`
  - `dem_sources/catalog.py`
- I/O layer:
  - `io/rasterio_io.py`
  - `io/metadata.py`
- Tests and docs:
  - `tests/`
  - `docs/`

Module interaction for model resolution and loading:

`CLI -> model_registry.resolve_model(...) -> cache/paths.model_path(...) -> download -> checksums.verify_sha256(...) -> engine.load(model_path)`

Module interaction for cache lifecycle and controls:

`CLI (cache commands) -> cache/reporting + cache/lifecycle -> cache/paths + cache/policy`

Module interaction for DEM fetch (current + planned):

`CLI (--fetch-hrdem) -> dem_sources/hrdem_stac -> I/O and preprocessing pipeline`

`Future: CLI/source selector -> dem_sources/catalog -> selected dem_sources/* backend`

## Consequences

- Runtime backend and CLI contract are decoupled.
- Testing boundaries are clearer by module responsibility.
- Future execution provider work stays isolated to engine components.
- Cache policy and lifecycle are centralized in a reusable subsystem rather than CLI-adjacent utilities.
