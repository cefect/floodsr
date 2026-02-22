# ADR-0001: Repository and Module Architecture

 
The codebase should keep the CLI thin, isolate model concerns from engine/runtime concerns, and preserve a swappable execution backend boundary.

 

see also:
- `docs/dev/adr/0011-parameters.md`

## module structure

 

- CLI surface:
  - `cli.py`
  - command routing for `infer`, `models`, `doctor`, `cache`
- Cache subsystem:
  - `cache/paths.py`
  - `cache/policy.py`
  - `cache/lifecycle.py`
  - `cache/reporting.py`
- Model subsystem:
  - `model_registry.py`
  - `models.json`
  - `models/base.py` (`class Model`)
  - `models/<model_version>.py` (e.g., `models/4690176_0_1770580046_train_base_16.py`)
  - model contracts per `0005-model-registry.md`
- Engine subsystem:
  - `engine/base.py`
  - `engine/ort.py`
  - `engine/providers.py`
 
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

## CLI org
main CLI commands/positional arguments:
- `floodsr infer`: main runner/function.  see below.
- `floodsr models`: model registry interactions. see `0005-model-registry.md`
- `floodsr doctor` 
- `floodsr cache`. see `0012-cache-policy-and-lifecycle.md`

each of these should have their own help sub-menu. 
with some global kwargs (mostly logging?)


## inference
requirements:
- needs to support multiple models with different I/O contracts, but the same CLI entrypoint and engine abstraction. see `0005-model-registry.md` and `0015-engine-runtime.md`
- `infer()` should create and teardown model workers via context management (`with ...`).
- needs to be **just push go** ready with a single command, but also support more advanced use cases (e.g., custom output path, custom model version, etc.)
- no `truth` or `metrics`. just inference
- provide progress bar and final diagnostics on completion (runtime, shape in, shape out, model version used, file size out)
- default output should be in cwd with the same name (and filetype and properties... other than shape) as the input but with `_sr.tif` suffix. allow `--out` to specify a different path.
 

### inference workflow
Under the hood, should implement a workflow like:
- resolve model artifact and model worker
  - if `--model-version` not specified, use first listed in `models.json` if found in cache, otherwise fallback to first in cache.  if nothing in cache, error with instructions to fetch a model.
- instantiate model worker from the resolved version (subclass of `Model`) and run it under context management.
- select engine runtime/provider policy per `0015-engine-runtime.md` (owned by model worker internals)
- platform pre-processing: general data conformance checks and corrections (e.g., reprojection, nodata handling, bbox, etc.). see `0009-preproccessing.md`
- **platform-model boundary**
- model super resolution: not all models will require all these. 
  - model specific pre-processing  (nice for interpolation work that can be applied raster wide efficnetly). 
  - tiling/windowing
  - **model-engine boundary**. see `0005-model-registry.md`
  - core inference
  - mosaicking/stitching
  - model specific post-processing  (e.g., de-normalization)
- final diagnostics, reporting, output writing, and cleanup.

In summary:
fetch model -> create model worker -> `with model_worker: model_worker.run(...)` -> output
