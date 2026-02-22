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

 