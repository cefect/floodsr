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
- either `--dem` or `--fetch-HRDEM` must be provided, but not both.
- add optional `--fetch-out` (or just `-fo`) flag to specify output path for fetched HRDEM tile. If not provided, the fetched tile will be cached in a temp directory (e.g. using `tempfile` module). so if `floodsr` with `--fetch-hrdem` is run multiple times with the same tile, it will reuse the cached version instead of fetching again.

### implementation strategy (agnostic internals, explicit CLI)
- keep CLI explicit and hard-coded to HRDEM for now (`--fetch-hrdem`).
- implement HRDEM as one backend under a backend-agnostic namespace:
  - `floodsr/dem_sources/base.py`
  - `floodsr/dem_sources/hrdem_stac.py`
  - `floodsr/dem_sources/catalog.py` (optional registry for future backends)
- this allows future  alternate backends without restructuring CLI flow.

### STAC entry point parameter placement
- store HRDEM STAC defaults in `floodsr/dem_sources/hrdem_stac.py` as module-scoped constants.
- keep these transparent by logging resolved source config at fetch start:
  - source id
  - STAC URL
  - collection
  - asset key
 


### proposed implementation
- retrieve bbox and crs from lores depth
- query STAC to identify intersection. if no intersection found, throw an error.
- if intersection found, fetch the corresponding HRDEM tile(s) and mosaic if multiple tiles intersect
- STAC entrypoint info should live in source-module constants (`dem_sources/hrdem_stac.py`)


#### post and pre processing changes
- fetched HRDEM tile reprojected to match the low-res depth raster crs and bbox
- then the `docs/dev/adr/0009-preproccessing.md` pre-processing steps and checks are applied to the fetched HRDEM tile  




## example snippets from other projects (need to be adapted):


### 2407_FHIMP._02_hrdem
`/home/cefect/LS/09_REPOS/02_JOBS/2407_FHIMP/smk/scripts/_02_hrdem.py`

import differneces between this and `floodsr`:
- this fetches the `hrdem-lidar` collection, which is structured as individual tiles, rather than the `hrdem-mosaic` collection, which is a single mosaic product.
-  this uses a pre-built tile index to retrieve info and extents, where `floodsr` simply fetches based on extents of lores depth. 


```python

import geopandas as gpd
from pystac_client import Client
import rasterio
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
 
from floodsr.io.rasterio_io import GEOTIF_OPTIONS



def assert_tile_index_chunk(tile_index_chunk):
    """Validate a single-tile index chunk."""
    if not __debug__: return 
    assert tile_index_chunk is not None, "Tile index chunk is None"
    assert len(tile_index_chunk) > 0, "Tile index chunk is empty"
    assert isinstance(tile_index_chunk,gpd.GeoDataFrame), (
        f"Unexpected chunk type: {type(tile_index_chunk)}"
    )
    assert "geometry" in tile_index_chunk, "Tile index chunk missing 'geometry' column"

    # enforce GeoDataFrame with a valid CRS and single tile


    assert tile_index_chunk.crs is not None, "Tile index chunk missing CRS"
    assert len(tile_index_chunk) == 1, f"Expected one tile per chunk, got {len(tile_index_chunk)}"
    assert tile_index_chunk.index.is_unique, "Tile index chunk has duplicate index entries"
    assert tile_index_chunk.geometry.notna().all(), "Tile index chunk has missing geometries"
    return tile_index_chunk

def assert_hrdem_output(out_fp, expected_crs=epsg_id, expected_nodata=-9999):
    """Validate the written HRDEM raster."""
    if not __debug__: return 
    out_fp = str(out_fp)
    assert os.path.exists(out_fp), f"HRDEM output not found: {out_fp}"
    with rasterio.open(out_fp) as ds:
        assert ds.count >= 1, "HRDEM output has no bands"
        assert ds.width > 0 and ds.height > 0, "HRDEM output has invalid dimensions"
        assert ds.crs == CRS.from_user_input(expected_crs), f"CRS mismatch: {ds.crs} != {expected_crs}"
        assert ds.nodata == expected_nodata, f"Nodata mismatch: {ds.nodata} != {expected_nodata}"
    return out_fp

def _stac_urls_for_aoi(asset_key, aoi_gdf, logger=None):
    """
    Query STAC for items intersecting the AOI (GeoDataFrame with CRS).

    STAC expects WGS84 (EPSG:4326) for spatial filters, so we reproject.
    """
    log = logger or logging.getLogger(__name__)
    client = Client.open(STAC_URL)
    aoi_4326 = aoi_gdf.to_crs(4326)
    minx, miny, maxx, maxy = aoi_4326.total_bounds

    search = client.search(
        collections=[COLLECTION],
        bbox=[minx, miny, maxx, maxy],
    )
    items = list(search.items())
    if not items:
        raise RuntimeError("No STAC items found intersecting AOI")

    urls = []
    for item in items:
        assets = item.assets
        if asset_key in assets:
            urls.append(assets[asset_key].href)

    urls = list(dict.fromkeys(urls))  # de-duplicate, preserve order
    if not urls:
        raise RuntimeError(f"No '{asset_key}' assets found in intersecting items")
    log.info(f"Found {len(urls)} {asset_key} asset(s)")
    return urls


def main_02_hrdem(tile_index_chunk, out_fp, asset=DEFAULT_ASSET, 
                  num_threads=1, debug=False, logger=None):
    """Fetch HRDEM assets intersecting the tile_index_chunk and mosaic them.
    

    Output
    ------
    out_fp : Path
        GeoTIFF file path for the output HRDEM mosaic clipped to the tile chunk bounds.
    """


    # --------------------
    # ----- defaults -----
    # --------------------
    log = logger or logging.getLogger('main_02_hrdem')
    asset = asset or DEFAULT_ASSET

    out_fp = Path(out_fp)

    log.info(f'Starting HRDEM fetch for tile chunk, output: {out_fp}')
    # Configure GDAL internal threading for raster I/O & compression.
    # This relies on the GDAL_NUM_THREADS environment variable set by
    # configure_gdal_threads(), which is honored by the GTiff driver.
    num_threads = configure_gdal_threads(num_threads, logger=log)

 
    # ----------------
    # ----- load -----
    # ----------------
    

    tile_label = tile_index_chunk.index[0] if len(tile_index_chunk.index) else "tile"
    log.debug(f"Tile chunk for {tile_label} threads = {num_threads}")

    assert_tile_index_chunk(tile_index_chunk)

    log.info("Querying STAC for intersecting assets...")
    urls = _stac_urls_for_aoi(asset, tile_index_chunk, logger=log)

    # COG mosaics for hrdem-lidar are in EPSG:3979 (Lambert conf. conic)
    # per HRDEM spec, so we can use AOI bounds directly in that CRS.
    tile_chunk_3979 = tile_index_chunk.to_crs(3979)
    minx, miny, maxx, maxy = tile_chunk_3979.total_bounds

    srcs = []
    try:
        for href in urls:
            log.info(f"Opening {href}")
            # GDAL_NUM_THREADS influences multi-threaded block I/O
            srcs.append(rasterio.open(href))

        # Pick minimum pixel size among sources to avoid upsampling
        try:
            res_x = min(abs(s.res[0]) for s in srcs if s.res)
            res_y = min(abs(s.res[1]) for s in srcs if s.res)
            res = (res_x, res_y)
        except ValueError:
            res = None

        nodatas = [s.nodata for s in srcs if s.nodata is not None]
        src_nodata = nodatas[0] if nodatas else None
        target_nodata = -9999.0

        mosaic, transform = merge(
            srcs,
            bounds=(minx, miny, maxx, maxy),
            res=res,
            nodata=src_nodata,
            method="first",
        )

        # Normalize dtype and nodata
        if mosaic.dtype != np.float32:
            mosaic = mosaic.astype("float32", copy=False)
        if src_nodata is not None and src_nodata != target_nodata:
            mosaic = np.where(mosaic == src_nodata, target_nodata, mosaic)
        elif src_nodata is None:
            # leave as-is; downstream nodata will be set to target_nodata
            pass

        meta_crs = srcs[0].crs
        if meta_crs is None:
            raise ValueError("Source CRS missing in HRDEM inputs")

        # Reproject to proj_crs if needed
        if meta_crs != proj_crs:
            log.debug(f"Reprojecting from {meta_crs} to {proj_crs}")
            bounds = array_bounds(mosaic.shape[1], mosaic.shape[2], transform)
            dst_transform, dst_width, dst_height = calculate_default_transform(
                meta_crs, proj_crs, mosaic.shape[2], mosaic.shape[1], *bounds
            )
            dst = np.empty((mosaic.shape[0], dst_height, dst_width), dtype="float32")
            for i in range(mosaic.shape[0]):
                reproject(
                    source=mosaic[i],
                    destination=dst[i],
                    src_transform=transform,
                    src_crs=meta_crs,
                    dst_transform=dst_transform,
                    dst_crs=proj_crs,
                    resampling=Resampling.bilinear,
                    dst_nodata=target_nodata,
                )
            mosaic = dst
            transform = dst_transform
            meta_crs = proj_crs

        meta = copy.deepcopy(GEOTIF_OPTIONS)
        assert meta["crs"] == f"EPSG:{epsg_id}", f"GEOTIF_OPTIONS CRS mismatch: {meta['crs']} != EPSG:{epsg_id}"
        meta.update(
            {
                "compress": meta.get("compress", "LZW"),
                "tiled": meta.get("tiled", True),
                "BIGTIFF": meta.get("BIGTIFF", "IF_SAFER"),
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "count": mosaic.shape[0],
                "nodata": target_nodata,
                "crs": meta_crs,
            }
        )
        log.debug(f"Output metadata:\n  {meta}")

        out_fp.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Writing mosaic to {out_fp}")

        # num_threads controls GTiff multi-threaded compression via GDAL_NUM_THREADS.
        with rasterio.open(out_fp, "w",num_threads=num_threads, **meta) as dst:
            dst.write(mosaic)

        log.info(f"Wrote {out_fp}")
        assert_hrdem_output(out_fp)

    finally:
        for s in srcs:
            s.close()

    log.info("HRDEM fetch complete.")
    return out_fp
