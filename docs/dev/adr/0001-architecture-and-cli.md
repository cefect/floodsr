# ADR-0001: Repository and Module Architecture

 
The codebase should keep the CLI thin, isolate model concerns from engine/runtime concerns, and preserve a swappable execution backend boundary.

 

see also:
- `docs/dev/adr/0011-parameters.md`

## module structure

 

- CLI surface:
  - `cli.py`
  - command routing for `tohr`, `models`, `doctor`, `cache`
- Cache subsystem:
  - `cache/paths.py`
  - `cache/policy.py`
  - `cache/lifecycle.py`
  - `cache/reporting.py`
- Model subsystem:
  - `model_registry.py`
  - `models.json`
  - `models/base.py` (`class Model`)
  - `models/<normalized_model_version>.py` (e.g., `models/ResUNet_16x_DEM.py`)
  - model contracts per `0005-model-registry.md`
- Pre-processing subsystem:
  - `preprocessing.py` (shared helpers)
  - pre-processing is internal to `tohr` and is not exposed as a standalone CLI entrypoint
- Engine subsystem:
  - `engine/base.py`
  - `engine/ort.py`
  - `engine/providers.py`
 
- Tiling subsystem:
  - `tiling.py` (single shared script for windowing + mosaicking)
- DEM source abstraction:
  - `dem_sources/base.py`
  - `dem_sources/hrdem_mosaic.py`
  - `dem_sources/catalog.py`
- I/O layer:
  - `io/rasterio_io.py`
  - `io/metadata.py`
- Tests and docs:
  - `tests/`
  - `docs/`

## CLI org
main CLI commands/positional arguments:
- `floodsr tohr`: main runner/function (to high resolution).  see below.
- `floodsr models`: model registry interactions. see `0005-model-registry.md`
- `floodsr doctor` 
- `floodsr cache`. see `0012-cache-policy-and-lifecycle.md`

main CLI global options:
- `floodsr --version`: prints the installed package version and exits before subcommand routing.

each of these should have their own help sub-menu. 
with some global kwargs (mostly logging?)

- User-facing docs and tutorial notebooks should demonstrate CLI usage as direct shell commands (`floodsr ...` in terminals and `!floodsr ...` in notebooks) rather than hiding CLI calls behind Python `subprocess` wrappers.
- `doctor` is the diagnostic surface for runtime metadata. In addition to dependency/provider diagnostics, it should report the installed `floodsr` package version and may report the loaded module path when needed to debug stale editable installs.
- The CLI version surface should report installed package metadata, not branch names or ad hoc Git descriptions.
- Explicit user output paths are authoritative and must not be silently rewritten to a different suffix or format.
- When an execution mode produces a mode-specific artifact type (for example, a tiled/windowed VRT assembly), the CLI/API should validate the explicit output path early and fail with a clear error rather than mutating the user path.


## CLI: tohr
requirements:
- needs to support multiple models with different I/O contracts, but the same CLI entrypoint and engine abstraction. see `0005-model-registry.md` and `0015-engine-runtime.md`
- `tohr` pipeline function should create and teardown model workers via context management (`with ...`).
- needs to be **just push go** ready with a single command, but also support more advanced use cases (e.g., custom output path, custom model version, etc.)
- should accept an alternate machine-interface JSON of CLI-equivalent parameters (e.g., model version and tiling settings). should use the same parameter schema as CLI args  
- no `truth` or `metrics`. just inference
- provide progress bar and final diagnostics on completion (runtime, shape in, shape out, model version used, file size out)
- CLI and Python APIs should use positive progress naming with `show_progress=True` / `--show-progress` as the default interface
- default output should be in cwd with the same name (and filetype and properties... other than shape) as the input but with `_sr.tif` suffix. allow `--out` to specify a different path.
 

### tohr workflow
Under the hood, should implement a workflow like:
- 1. resolve model artifact and model worker
  - if `--model-version` not specified, use first listed in `models.json` if found in cache, otherwise fallback to first in cache.  if nothing in cache, error with instructions to fetch a model.
- 2. optional DEM fetch.
- 3. run platform pre-processing inside the `tohr` workflow via shared helpers in `floodsr/preprocessing.py` to produce platform-model boundary artifacts. see `0009-preproccessing.md`.
  - keep two internal execution paths:
    - a simple in-memory path for modest extents
    - a raster-backed windowed path for memory-constrained large extents
- 4. instantiate model worker from the resolved version (subclass of `Model`) and execute model-specific super-resolution via `with ...: worker.run(...)` on the pre-processed platform-model boundary artifacts (not raw user rasters).
  - select engine runtime/provider policy per `0015-engine-runtime.md` (owned by model worker internals)
  - model workers must call shared tiling utilities from `tiling.py`.
  - the raster-backed windowed pipeline is restricted to hard-window inference in this phase; feathered blending remains on the simple path
- 5. final model post-processing and output materialization.
- 6. diagnostics/reporting/cleanup.

In summary:
resolve model -> internal preprocess -> `with model_worker: model_worker.run(...)` -> write output -> diagnostics
