# ADR - 0000: scope of MVP and future

features/scope of MVP:
- CLI focused Super-Resolution for flood hazard rasters
- Ingests lores water grid and hires DEM and infers a hires water grid using the specifeid model and backend.
- fetches model weights and params from a manifest and caches them locally
- ONNX Runtime w/ CPUExecutionProvider. 
- pip installable package with CLI entrypoint

future features:
- QGIS plugin GUI (separate project)
- GPU support

Out of scope:
- performance/validation against a known tile (only do a bit of this in dev for pytests)
- plotting/visualization of results (a tiny bit for dev only)

## CLI

### infer
- no `truth` or `metrics`. just inference
- if `--model-version` not specified, use first listed in `models.json` if found in cache, otherwise fallback to first in cache.  if nothing in cache, error with instructions to fetch a model.
- provide progress bar and final diagnostics on completion (runtime, shape in, shape out, model version used, file size out)
