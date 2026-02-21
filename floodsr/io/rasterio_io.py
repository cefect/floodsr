"""Rasterio-specific I/O defaults for writing rasters."""

# Default GeoTIFF write options used by raster outputs.
GEOTIF_OPTIONS = {
    "driver": "GTiff",
    "dtype": "float32",
    "compress": "LZW",
    "nodata": -9999,
}


def get_geotif_options() -> dict:
    """Return a copy of default GeoTIFF options for safe per-call mutation."""
    return dict(GEOTIF_OPTIONS)
