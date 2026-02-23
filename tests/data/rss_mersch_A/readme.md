# RSS mersch A tile

## Provenance

Prepared by `bin/clip_test_grids.sh` from:
- `_inputs/RSSHydro/mersch/032/ResultA.tif` (lowres depth)
- `_inputs/RSSHydro/mersch/002/ResultA.tif` (hires depth)
- `_inputs/RSSHydro/mersch/002/DEM.tif` (hires DEM)

## Outputs
- `lowres030.tif`: 256x256, 30m, EPSG:2169
- `hires002.tif`: 3840x3840, 2m, EPSG:2169
- `hires002_dem.tif`: 3840x3840, 2m, EPSG:2169

## Notes
- Lowres CRS is explicitly set from the DEM CRS.
- Hires depth and DEM are clipped to the lowres bbox using nearest-neighbor resampling at fixed 2m resolution for pixel alignment.
- GeoTIFF driver options are set from `floodsr/io/rasterio_io.py`: `GTiff`, `Float32`, `LZW`, `NoData=-9999`.
- This case is not directly compatible with current ToHR model tests because the current ONNX model contract expects a `32x32` lowres input tile.
