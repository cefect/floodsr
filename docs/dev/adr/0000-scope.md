# ADR - 0000: scope of MVP and future

features/scope of MVP:
- CLI focused Super-Resolution for flood hazard rasters
- Ingests lores water grid and hires DEM and infers a hires water grid using the specifeid model and backend.
- fetches model weights and params from a manifest and caches them locally
- ONNX Runtime w/ CPUExecutionProvider. 
- pip installable package with CLI entrypoint
- let rasterio handle raster driver support. i.e., default to GeoTiff, but allow any rasterio-supported format as input/output.

future features:
- QGIS plugin GUI (separate project)
- GPU support

Out of scope:
- runtime performance/validation against a known tile
- plotting/visualization of results (a tiny bit for dev only)

## CLI

### infer
- no `truth` or `metrics`. just inference
- if `--model-version` not specified, use first listed in `models.json` if found in cache, otherwise fallback to first in cache.  if nothing in cache, error with instructions to fetch a model.
- provide progress bar and final diagnostics on completion (runtime, shape in, shape out, model version used, file size out)
- default output should be in cwd with the same name (and filetype and properties... other than shape) as the input but with `_sr.tif` suffix. allow `--out` to specify a different path.
