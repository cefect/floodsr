# RSS Dudelange test case

## Provenance

Cropped and prepped with `bin/clip_test_grids.sh`.

## Notes

- CRS is set to EPSG:2169.
- The large-windowed CostGrow regression derives a temporary variant at test runtime by duplicating the base rasters horizontally.
- That derived variant exists only to push CostGrow over the large-raster `windowed + hard` threshold while keeping the committed fixture set small.
