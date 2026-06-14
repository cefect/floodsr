# ADR: HRDEM fetching
Rather than rely on the user specifying the DEM, this new `floodsr` feature will optionally fetch an equivalent tile from the [HRDEM mosaic](https://open.canada.ca/data/en/dataset/0fe65119-e96e-4a57-8bfe-9d9245fba06b).

Quote on HRDEM mosaic:
```
Unlike the HRDEM product in the same series, which is distributed by acquisition project without integration between projects, the mosaic is created to provide a single, continuous representation of strategy data. The most recent datasets for a given territory are used to generate the mosaic. 
```

STAC entry point:
```python
STAC_URL = "https://datacube.services.geo.ca/api"
COLLECTION = "hrdem-mosaic-1m"   
DEFAULT_ASSET = "dtm"
```

## proposed implementation
### CLI behavior
- make `--dem` optional
- add optional `--fetch-hrdem` (or just `-f`) flag to trigger HRDEM fetch instead. 
- either `--dem` or `--fetch-hrdem` must be provided, but not both.
- add optional `--fetch-out`   flag to specify output path for fetched HRDEM tile. If not provided, the fetched tile will live in the temp directory (e.g. using `tempfile` module) NOT the cache. should provide some lazy caching so if the same tile is requested in the same fetch session, it doesn't re-fetch from the source. 
  - per `0001-architecture-and-cli.md`, do not silently rewrite an explicit user path.
  - if the fetch resolves to the tiled/VRT path and the user provides an explicit `--fetch-out`, the path must already end with `.vrt`.
  - if tiled fetch is selected and the user passes a non-`.vrt` explicit path, fail early with a clear error.
  - if `--fetch-out` is omitted, the implementation may still choose the default suffix based on fetch mode (`.tif` for rapid/non-windowed, `.vrt` for tiled/windowed).

### implementation strategy (agnostic internals, explicit CLI)
- keep CLI explicit and hard-coded to HRDEM (`--fetch-hrdem`).
- implement HRDEM as one backend under a backend-agnostic namespace:
  - `floodsr/dem_sources/base.py`
  - `floodsr/dem_sources/hrdem_mosaic.py`
  - `floodsr/dem_sources/catalog.py`

### entry point parameter placement
- store HRDEM STAC defaults in `floodsr/dem_sources/hrdem_mosaic.py` as module-scoped constants.
- keep these transparent by logging resolved source config at fetch start 
- DEM fetch owns its own large-raster tiling implementation, but it should build that implementation from shared helpers in `floodsr/tiling.py` per `ADR-0008`.
 
### strategy/constraints for HRDEM Mosaic
- minimize server-side processing:
  - always query in mosaic native CRS (3979)
  - fetch in native resolution (detect this)
- fetching can be slow
  - for large extents, pre-tile and fetch in chunks
  - chunks should live in cache to enable re-use and restart
  - bundle chunks into a VRT for downstream processing (avoid in-memory merge)
- HRDEM coverage is irrugalr and discontinuous and dynamic:
  - pre-filter the fetch by querying [HRDEM Project Extent](https://maps-cartes.services.geo.ca/server_serveur/rest/services/NRCan/coverage_HRDEM_en/MapServer/4). see `proof_of_concepts/hrdem_project_extent.ipynb`
  - use returned project extent polygon geometries for tile filtering (not feature bbox-only checks)
  - warn when the user attempts to query a fetch tile with no coverage, and omit fetching this tile. 
- Use 3979 bbox only for candidate discovery (STAC), then exact intersection

### proposed implementation
- retrieve bbox (4326) and footprint polygon (3979) from lores tile exents. 
- query STAC with bbox (4326) to identify assets. if no intersection found, throw an error.
- estimate memory of fetch and split workflow into shared execution styles based on a threshold:
- `simple`:
    - fetch from hrdem mosaic in one go (no tiling) and merge in-memory. (current implementation)
- `windowed`:
  - download HRDEM Project Extent features intersecting fetch footprint polygon (3979). throw error if no features returned.
  - build a tiling scheme in fetch-native/source CRS using source-space pixel windows (not lores pixel windows). if source resolution metadata is unavailable, assume 1m.
  - fetch tiles should align to source pixels/resolution. they do not need to align to lores pixel edges.
  - inside the tiled fetch loop, convert each fetch-tile geometry/bounds to 3979 only for HRDEM Project Extent polygon intersection checks.
    - throw error if no intersecting tiles
    - warn when a tile does not intersect and write nodata for that tile (skip source reads for that tile).
  - loop through the tile grid (progress bar)
    - bbox fetch tile to 3979
    - build a cache payload and filepath for this tile
    - return cache if it exists, otherwise proceed to download
    - read/download mosaic cog for this tile
    - write to cache as a chunked GeoTIFF (see below tile write defaults)
    - build a VRT over the fetched tiles for downstream processing (avoid in-memory merge)
  - let gdal/backends handle multi-processing/threading for now (keep our implementation simple). 

see `dev/proof_of_concepts/hrdem_fetch.ipynb`

#### tile write defaults
```bash
TILED=YES
BLOCKXSIZE=512
BLOCKYSIZE=512
COMPRESS=LZW
PREDICTOR=3
NUM_THREADS=ALL_CPUS
BIGTIFF=IF_SAFER
```
 
### decision update (memory handling)
- use windowed/on-disk DEM fetch processing to avoid full in-memory merge for large extents.
- keep HRDEM fetch at native source resolution in this phase (no fetch-resolution parameter).
- build and use a VRT to stitch fetched chunks/tiles into one virtual mosaic.
- add early diagnostics/guards for oversized fetches so failures are explicit (not silent OOM kills).
- DEM fetch must implement the shared mosaicking vocabulary from `ADR-0008`:
  - `hard` (no `feather` for DEM fetching)
- For HRDEM, the current baseline is:
  - `simple + hard`
  - `windowed + hard`

### decision update (explicit output path contract)
- explicit user output paths are authoritative and must not be silently rewritten. this is a general CLI/API rule; see `0001-architecture-and-cli.md`.
- tiled/windowed fetches produce a VRT assembly artifact, so explicit tiled outputs must use a `.vrt` suffix.
- non-windowed fetches produce a GeoTIFF artifact, so explicit non-windowed outputs must use a raster filepath appropriate for that mode.
- we do NOT want to "honor explicit tiled `.tif` outputs by materializing the VRT to a GeoTIFF".
- this keeps the tiled fetch implementation simple and transparent about its actual output artifact, rather than adding an implicit format-conversion step.
 


#### post and pre processing changes
- treat the incoming HRDEM asset (should always be 3979) identical to explicitly user-provided DEMs.
  - throw an early verbose error if `--crs-policy=strict` and the incoming lores is not 3979.
- keep `crs_policy` behavior in preprocessing/runtime, not in the fetch tiling logic.
  - `strict`: require identical CRS after fetch as usual; do not auto-reproject in the fetch layer.
  - `use-dem`: fetched HRDEM CRS remains the downstream target CRS.
  - `use-lores`: fetched HRDEM remains in source CRS until preprocessing reprojects it to the lores CRS.
- then the `docs/dev/adr/0009-preproccessing.md` pre-processing steps and checks are applied to the fetched HRDEM tile  


 
