# ADR-0004: Repository and Module Architecture

 
The codebase should keep the CLI thin, isolate engine/runtime concerns, and preserve a swappable execution backend boundary.

## Decision

Use a structure centered on these modules:

- CLI surface:
  - `cli.py`
  - `model_registry.py`
  - `cache_paths.py`
  - `checksums.py`
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
- I/O layer:
  - `io/rasterio_io.py`
  - `io/npy_io.py`
  - `io/metadata.py`
- Tests and docs:
  - `tests/`
  - `docs/`

Module interaction for model resolution and loading:

`CLI -> model_registry.resolve_model(...) -> cache_paths.model_path(...) -> download -> checksums.verify_sha256(...) -> engine.load(model_path)`

## Consequences

- Runtime backend and CLI contract are decoupled.
- Testing boundaries are clearer by module responsibility.
- Future execution provider work stays isolated to engine components.

