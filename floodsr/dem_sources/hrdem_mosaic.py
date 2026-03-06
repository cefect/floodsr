"""HRDEM STAC backend implementation."""

import hashlib
import json
import logging
import shutil
import tempfile
from functools import partial
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode
from urllib.request import urlopen

from osgeo import gdal

import geopandas as gpd
import numpy as np

import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, transform_bounds
from rasterio.windows import Window, bounds as window_bounds, from_bounds as window_from_bounds
from shapely.geometry import shape
from tqdm.auto import tqdm


from floodsr.dem_sources.base import DemFetchResult


SOURCE_ID = "hrdem"
STAC_URL = "https://datacube.services.geo.ca/api"
COLLECTION = "hrdem-mosaic-1m"
DEFAULT_ASSET = "dtm"
PROJECT_EXTENT_URL = (
    "https://maps-cartes.services.geo.ca/server_serveur/rest/services/NRCan/coverage_HRDEM_en/MapServer/4"
)
DEFAULT_FETCH_WINDOW_SIZE = 2048
DEFAULT_FETCH_MEMORY_LIMIT_GIB = 16.0
DEFAULT_STAC_QUERY_LIMIT = 200
DEFAULT_BOUNDS_DENSIFY_PTS = 21
DEFAULT_PROJECT_EXTENT_TIMEOUT_S = 60.0
DEFAULT_TILE_COMPRESS = "LZW"
DEFAULT_TILE_PREDICTOR = 3
DEFAULT_TILE_BLOCK_SIZE = 512
DEFAULT_TILE_NUM_THREADS = "ALL_CPUS"
DEFAULT_TILE_BIGTIFF = "IF_SAFER"
DEFAULT_VRT_RESOLUTION = "highest"
DEFAULT_WORK_NODATA = np.float32(-3.4028235e38)
MIN_TILED_BLOCK_SIZE = 16
TILE_CACHE_DIR_NAME = "floodsr_hrdem_tile_cache"
TEMP_OUTPUT_PREFIX = "floodsr_hrdem_output"


def _build_request_token(
    depth_bounds: tuple[float, float, float, float],
    stac_url: str,
    collection: str,
    asset_key: str,
    tiling_enabled: bool,
    fetch_window_size: int,
    use_project_extent_filter: bool,
) -> str:
    """Build one stable request token used for output naming and tile-cache namespacing."""
    bounds_token = ",".join(f"{float(v):.8f}" for v in depth_bounds)
    payload = (
        f"{bounds_token}|{stac_url}|{collection}|{asset_key}|"
        f"tiling={int(bool(tiling_enabled))}|window={int(fetch_window_size)}|"
        f"project_extent={int(bool(use_project_extent_filter))}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _resolve_default_output_path(request_token: str, suffix: str) -> Path:
    """Resolve one temporary output filepath when caller does not pass output_fp."""
    return (Path(tempfile.gettempdir()) / f"{TEMP_OUTPUT_PREFIX}_{request_token}{suffix}").resolve()


def _build_tile_cache_key(
    request_token: str,
    tile_bounds: tuple[float, float, float, float],
    tile_shape: tuple[int, int],
    source_crs,
    asset_hrefs: list[str],
) -> str:
    """Build one stable cache key for a fetched GeoTIFF tile."""
    bounds_token = ",".join(f"{float(v):.8f}" for v in tile_bounds)
    shape_token = f"{int(tile_shape[0])}x{int(tile_shape[1])}"
    crs_token = source_crs.to_string() if hasattr(source_crs, "to_string") else str(source_crs)
    asset_token = "|".join(str(v) for v in asset_hrefs)
    payload = f"{request_token}|bounds={bounds_token}|shape={shape_token}|crs={crs_token}|assets={asset_token}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _resolve_tile_cache_fp(cache_key: str) -> Path:
    """Resolve cache filepath for one GeoTIFF tile."""
    cache_dir = (Path(tempfile.gettempdir()) / TILE_CACHE_DIR_NAME).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (cache_dir / f"{cache_key}.tif").resolve()


def _materialize_tif_with_cache(
    out_fp: Path,
    use_cache: bool,
    cache_key: str,
    writer: Callable[[Path], None],
    logger=None,
) -> tuple[Path, bool]:
    """Materialize one GeoTIFF tile to output, reading/writing cache when enabled."""
    log = logger or logging.getLogger(__name__)
    cache_fp = _resolve_tile_cache_fp(cache_key)
    if use_cache and cache_fp.exists():
        if out_fp != cache_fp:
            out_fp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache_fp, out_fp)
        log.debug(f"tile cache hit:\n    {cache_fp}")
        return out_fp, True

    write_target_fp = cache_fp if use_cache else out_fp
    write_target_fp.parent.mkdir(parents=True, exist_ok=True)
    writer(write_target_fp)
    if use_cache and out_fp != cache_fp:
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache_fp, out_fp)
    log.debug(f"tile cache miss; wrote:\n    {write_target_fp}")
    return out_fp, False


def _raster_has_any_valid_pixels(dem_fp: Path, dst_nodata: float) -> bool:
    """Return True when the raster contains at least one non-nodata pixel."""
    with rasterio.open(dem_fp) as ds:
        arr = ds.read(1)
    if np.isnan(float(dst_nodata)):
        valid_mask = ~np.isnan(arr)
    else:
        valid_mask = ~np.isclose(arr, np.float32(dst_nodata))
    return bool(valid_mask.any())


def _open_source_meta_list(asset_hrefs: list[str], expected_crs) -> list[dict[str, object]]:
    """Open source rasters once and validate one shared CRS."""
    src_meta_l: list[dict[str, object]] = []
    for href in asset_hrefs:
        src_ds = rasterio.open(href)
        src_crs = src_ds.crs
        assert src_crs is not None, f"asset CRS is required: {href}"
        if src_crs != expected_crs:
            src_ds.close()
            for src_meta in src_meta_l:
                src_meta["ds"].close()
            raise AssertionError(
                f"all HRDEM assets must share one CRS without auto-reprojection: {src_crs} != {expected_crs} for {href}"
            )
        src_meta_l.append({"href": href, "ds": src_ds, "bounds": tuple(float(v) for v in src_ds.bounds)})
    return src_meta_l


def _close_source_meta_list(src_meta_l: list[dict[str, object]]):
    """Close source dataset handles opened by _open_source_meta_list."""
    for src_meta in src_meta_l:
        src_meta["ds"].close()


def _build_valid_mask(arr: np.ndarray, src_nodata: float | None, work_nodata: float | None = None) -> np.ndarray:
    """Build one valid-pixel mask from optional work and source nodata semantics."""
    valid_mask = np.ones(arr.shape, dtype=bool)
    if work_nodata is not None:
        valid_mask &= ~np.isclose(arr, np.float32(work_nodata))
    if src_nodata is not None:
        if np.isnan(float(src_nodata)):
            valid_mask &= ~np.isnan(arr)
        else:
            valid_mask &= ~np.isclose(arr, np.float32(src_nodata))
    return valid_mask


def _finalize_float32_tile(arr: np.ndarray, valid_mask: np.ndarray, dst_nodata: float) -> np.ndarray:
    """Write-ready float32 tile where invalid pixels are replaced by destination nodata."""
    return np.where(valid_mask, arr, np.float32(dst_nodata)).astype(np.float32, copy=False)


def _write_window_tile(
    write_fp: Path,
    tile_profile: dict[str, object],
    window_src_meta_l: list[dict[str, object]],
    fetch_bounds: tuple[float, float, float, float],
    win_height: int,
    win_width: int,
    dst_nodata: float,
    work_nodata: float,
    tile_status_d: dict[str, bool],
):
    """Write one fetched window tile from intersecting source rasters."""
    merged_win = np.full((win_height, win_width), np.float32(dst_nodata), dtype=np.float32)
    if window_src_meta_l:
        merged_work = np.full((win_height, win_width), work_nodata, dtype=np.float32)
        valid_mask = np.zeros((win_height, win_width), dtype=bool)
        for src_meta in window_src_meta_l:
            src_ds = src_meta["ds"]
            src_window = window_from_bounds(*fetch_bounds, transform=src_ds.transform)
            read_win = src_ds.read(
                1,
                window=src_window,
                out_shape=(win_height, win_width),
                resampling=Resampling.nearest,
                boundless=True,
                fill_value=float(work_nodata),
            ).astype(np.float32, copy=False)
            current_valid = _build_valid_mask(read_win, src_ds.nodata, work_nodata=work_nodata)
            if current_valid.any():
                merged_work[current_valid] = read_win[current_valid]
                valid_mask |= current_valid
        if valid_mask.any():
            tile_status_d["has_valid"] = True
            merged_win = _finalize_float32_tile(merged_work, valid_mask, dst_nodata)
    with rasterio.open(write_fp, "w", **tile_profile) as tile_ds:
        tile_ds.write(merged_win, 1)


def _resolve_depth_query_geometry(
    depth_lr_fp: str | Path,
) -> dict[str, object]:
    """Read low-res raster geometry used for STAC query and output alignment."""
 


    depth_path = Path(depth_lr_fp).expanduser().resolve()
    assert depth_path.exists(), f"low-res depth raster does not exist: {depth_path}"
    with rasterio.open(depth_path) as depth_ds:
        depth_crs = depth_ds.crs
        depth_bounds = tuple(float(v) for v in depth_ds.bounds)
        depth_nodata = depth_ds.nodata
        depth_width = int(depth_ds.width)
        depth_height = int(depth_ds.height)
        depth_transform = depth_ds.transform
    assert depth_crs is not None, f"low-res depth CRS is required for STAC query: {depth_path}"

    # Translate low-res bounds to WGS84 for STAC query filtering.
    lowres_bbox_4326 = transform_bounds(
        depth_crs,
        "EPSG:4326",
        *depth_bounds,
        densify_pts=DEFAULT_BOUNDS_DENSIFY_PTS,
    )
    assert lowres_bbox_4326[0] < lowres_bbox_4326[2], f"invalid transformed bbox x ordering: {lowres_bbox_4326}"
    assert lowres_bbox_4326[1] < lowres_bbox_4326[3], f"invalid transformed bbox y ordering: {lowres_bbox_4326}"
    return {
        "depth_fp": depth_path,
        "depth_crs": depth_crs,
        "depth_bounds": depth_bounds,
        "depth_nodata": depth_nodata,
        "depth_width": depth_width,
        "depth_height": depth_height,
        "depth_transform": depth_transform,
        "bbox_4326": tuple(float(v) for v in lowres_bbox_4326),
    }


def _estimate_fetch_geometry(
    depth_crs,
    depth_bounds: tuple[float, float, float, float],
    reference_href: str,
) -> dict[str, object]:
    """Estimate output fetch grid geometry in source CRS using one reference asset."""
    with rasterio.open(reference_href) as first_ds:
        first_crs = first_ds.crs
        assert first_crs is not None, f"asset CRS is required: {reference_href}"
        source_nodata = first_ds.nodata
        first_res = (abs(float(first_ds.res[0])), abs(float(first_ds.res[1])))
        clip_bounds = transform_bounds(
            depth_crs,
            first_crs,
            *depth_bounds,
            densify_pts=DEFAULT_BOUNDS_DENSIFY_PTS,
        )

    assert clip_bounds[0] < clip_bounds[2], f"invalid transformed clip bounds x ordering: {clip_bounds}"
    assert clip_bounds[1] < clip_bounds[3], f"invalid transformed clip bounds y ordering: {clip_bounds}"
    pixel_width, pixel_height = first_res
    assert pixel_width > 0 and pixel_height > 0, f"invalid source resolution: {first_res}"
    est_width = max(1, int(np.ceil((clip_bounds[2] - clip_bounds[0]) / pixel_width)))
    est_height = max(1, int(np.ceil((clip_bounds[3] - clip_bounds[1]) / pixel_height)))
    est_pixels = int(est_width * est_height)
    est_float32_gib = (est_pixels * 4.0) / (1024.0**3)
    out_transform = from_bounds(*clip_bounds, est_width, est_height)
    return {
        "source_crs": first_crs,
        "source_nodata": source_nodata,
        "source_res": first_res,
        "clip_bounds": clip_bounds,
        "width": est_width,
        "height": est_height,
        "pixels": est_pixels,
        "float32_gib": est_float32_gib,
        "out_transform": out_transform,
    }


def _query_hrdem_assets(
    bbox_4326: tuple[float, float, float, float],
    stac_url: str,
    collection: str,
    asset_key: str,
    allow_empty: bool = False,
    stac_query_limit: int = DEFAULT_STAC_QUERY_LIMIT,
) -> tuple[list[str], list[str], list[object]]:
    """Query STAC and return intersecting item ids, asset hrefs, and STAC items."""
    from pystac_client import Client

    client = Client.open(stac_url)
    search = client.search(
        collections=[collection],
        bbox=list(bbox_4326),
        limit=int(stac_query_limit),
    )
    # Materialize the search result once so we can validate coverage and assets.
    items = list(search.items())
    if not items:
        if allow_empty:
            return [], [], []
        raise RuntimeError(
            f"HRDEM STAC query returned 0 items for bbox={bbox_4326} collection={collection} at {stac_url}"
        )

    item_ids: list[str] = []
    asset_hrefs: list[str] = []
    for item in items:
 
        if asset_key not in item.assets:
            continue
        href = item.assets[asset_key].href
        if href is None:
            continue
        item_ids.append(str(item.id))
        asset_hrefs.append(str(href))

    if not asset_hrefs:
        if allow_empty:
            return [], [], []
        raise RuntimeError(
            f"HRDEM STAC returned items but no '{asset_key}' assets for bbox={bbox_4326}"
        )
    return item_ids, asset_hrefs, items


def _bounds_intersect(
    left_bounds: tuple[float, float, float, float],
    right_bounds: tuple[float, float, float, float],
) -> bool:
    """Return True when two axis-aligned bounds overlap."""
    return not (
        left_bounds[2] <= right_bounds[0]
        or left_bounds[0] >= right_bounds[2]
        or left_bounds[3] <= right_bounds[1]
        or left_bounds[1] >= right_bounds[3]
    )


def _resolve_gtiff_write_options(width: int, height: int) -> dict[str, object]:
    """Build GTiff creation options with configurable tile defaults."""
    width = int(width)
    height = int(height)
    options_d: dict[str, object] = {
        "compress": DEFAULT_TILE_COMPRESS,
        "predictor": int(DEFAULT_TILE_PREDICTOR),
        "num_threads": DEFAULT_TILE_NUM_THREADS,
        "bigtiff": DEFAULT_TILE_BIGTIFF,
    }
    block_x = min(int(DEFAULT_TILE_BLOCK_SIZE), width)
    block_y = min(int(DEFAULT_TILE_BLOCK_SIZE), height)
    block_x = block_x - (block_x % MIN_TILED_BLOCK_SIZE)
    block_y = block_y - (block_y % MIN_TILED_BLOCK_SIZE)
    if block_x >= MIN_TILED_BLOCK_SIZE and block_y >= MIN_TILED_BLOCK_SIZE:
        options_d["tiled"] = True
        options_d["blockxsize"] = int(block_x)
        options_d["blockysize"] = int(block_y)
    return options_d


def _filter_hrdem_assets_to_clip_bounds(
    item_ids: list[str],
    asset_hrefs: list[str],
    clip_bounds: tuple[float, float, float, float],
    expected_crs,
    logger=None,
) -> tuple[list[str], list[str]]:
    """Filter STAC assets to exact raster-bound intersection with clip bounds."""
    log = logger or logging.getLogger(__name__)
    assert len(item_ids) == len(asset_hrefs), (
        f"item_ids and asset_hrefs must align one-to-one, got {len(item_ids)} and {len(asset_hrefs)}"
    )
    filtered_pairs_l: list[tuple[str, str]] = []
    for item_id, href in zip(item_ids, asset_hrefs):
        with rasterio.open(href) as src_ds:
            src_crs = src_ds.crs
            assert src_crs is not None, f"asset CRS is required: {href}"
            if src_crs != expected_crs:
                raise AssertionError(
                    f"all HRDEM assets must share one CRS without auto-reprojection: "
                    f"{src_crs} != {expected_crs} for {href}"
                )
            src_bounds = tuple(float(v) for v in src_ds.bounds)
        if _bounds_intersect(src_bounds, clip_bounds):
            filtered_pairs_l.append((str(item_id), str(href)))
    if filtered_pairs_l:
        filtered_item_ids, filtered_asset_hrefs = zip(*filtered_pairs_l)
        log.debug(
            f"exact-intersection filter retained {len(filtered_asset_hrefs):,}/{len(asset_hrefs):,} "
            "assets by raster bounds"
        )
        return list(filtered_item_ids), list(filtered_asset_hrefs)
    log.warning("exact-intersection filter retained 0 assets by raster bounds")
    return [], []


def _build_project_extent_gdf(
    clip_bounds_3979: tuple[float, float, float, float],
    project_extent_url: str = PROJECT_EXTENT_URL,
    timeout_s: float = DEFAULT_PROJECT_EXTENT_TIMEOUT_S,
    logger=None,
):
    """Download and parse HRDEM project extent features into one EPSG:3979 GeoDataFrame."""
    feature_l = _download_hrdem_project_extent_features(
        clip_bounds_3979,
        project_extent_url=project_extent_url,
        timeout_s=timeout_s,
        logger=logger,
    )
    if not feature_l:
        raise RuntimeError(f"no HRDEM project-extent features intersect clip bounds={clip_bounds_3979}")
    project_extent_gdf = gpd.GeoDataFrame.from_features(feature_l, crs="EPSG:3979")
    if project_extent_gdf.empty:
        raise RuntimeError("project extent query returned an empty GeoDataFrame")
    project_extent_gdf = project_extent_gdf[project_extent_gdf.geometry.notnull()].copy()
    if project_extent_gdf.empty:
        raise RuntimeError("project extent query returned features without usable geometry")
    return project_extent_gdf.reset_index(drop=True)


def _build_fetch_tile_grid_gdf(
    depth_width: int,
    depth_height: int,
    fetch_window_size: int,
    depth_transform,
    depth_crs,
):
    """Build one depth-native-CRS GeoDataFrame describing the full fetch tile grid."""
    fetch_window_size = int(fetch_window_size)
    depth_width = int(depth_width)
    depth_height = int(depth_height)
    tile_meta_l: list[dict[str, object]] = []
    tile_id = 0
    for row_off in range(0, depth_height, fetch_window_size):
        for col_off in range(0, depth_width, fetch_window_size):
            lowres_height = min(fetch_window_size, depth_height - row_off)
            lowres_width = min(fetch_window_size, depth_width - col_off)
            lowres_window = Window(col_off=col_off, row_off=row_off, width=lowres_width, height=lowres_height)
            lowres_bounds = window_bounds(lowres_window, depth_transform)
            win_geometry = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [float(lowres_bounds[0]), float(lowres_bounds[1])],
                        [float(lowres_bounds[0]), float(lowres_bounds[3])],
                        [float(lowres_bounds[2]), float(lowres_bounds[3])],
                        [float(lowres_bounds[2]), float(lowres_bounds[1])],
                        [float(lowres_bounds[0]), float(lowres_bounds[1])],
                    ]
                ],
            }
            tile_meta_l.append(
                {
                    "tile_id": int(tile_id),
                    "row_off": int(row_off),
                    "col_off": int(col_off),
                    "lowres_height": int(lowres_height),
                    "lowres_width": int(lowres_width),
                    "win_bounds": tuple(float(v) for v in lowres_bounds),
                    "geometry": shape(win_geometry),
                }
            )
            tile_id += 1
    return gpd.GeoDataFrame(tile_meta_l, geometry="geometry", crs=depth_crs)


def _resolve_fetch_tile_selection_mask(tile_grid_gdf, project_extent_gdf) -> np.ndarray:
    """Return a bool mask for grid tiles intersecting project extent polygons."""
    if tile_grid_gdf.empty:
        return np.zeros(0, dtype=bool)
    if project_extent_gdf.empty:
        return np.zeros(len(tile_grid_gdf), dtype=bool)
    extent_union = (
        project_extent_gdf.geometry.union_all()
        if hasattr(project_extent_gdf.geometry, "union_all")
        else project_extent_gdf.unary_union
    )
    if extent_union is None or extent_union.is_empty:
        return np.zeros(len(tile_grid_gdf), dtype=bool)
    return tile_grid_gdf.geometry.intersects(extent_union).to_numpy(dtype=bool)


def _download_hrdem_project_extent_features(
    clip_bounds_3979: tuple[float, float, float, float],
    project_extent_url: str = PROJECT_EXTENT_URL,
    timeout_s: float = DEFAULT_PROJECT_EXTENT_TIMEOUT_S,
    logger=None,
) -> list[dict[str, object]]:
    """Download HRDEM project-extent features that intersect one 3979 envelope."""
    log = logger or logging.getLogger(__name__)
    assert clip_bounds_3979[0] < clip_bounds_3979[2], f"invalid x ordering for clip_bounds_3979: {clip_bounds_3979}"
    assert clip_bounds_3979[1] < clip_bounds_3979[3], f"invalid y ordering for clip_bounds_3979: {clip_bounds_3979}"
    assert timeout_s > 0, f"timeout_s must be > 0, got {timeout_s}"
    query_url = f"{project_extent_url.rstrip('/')}/query"
    geometry_d = {
        "xmin": float(clip_bounds_3979[0]),
        "ymin": float(clip_bounds_3979[1]),
        "xmax": float(clip_bounds_3979[2]),
        "ymax": float(clip_bounds_3979[3]),
        "spatialReference": {"wkid": 3979},
    }
    params_d = {
        "where": "1=1",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": 3979,
        "outSR": 3979,
        "returnGeometry": "true",
        "outFields": "*",
        "f": "geojson",
        "geometry": json.dumps(geometry_d, separators=(",", ":")),
    }
    request_url = f"{query_url}?{urlencode(params_d)}"
    with urlopen(request_url, timeout=float(timeout_s)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "error" in payload:
        raise RuntimeError(f"project extent query failed: {payload['error']}")
    feature_l = payload.get("features", [])
    if not isinstance(feature_l, list):
        raise RuntimeError(f"project extent query returned invalid feature payload type: {type(feature_l)!r}")
    log.info(
        f"downloaded {len(feature_l):,} HRDEM project extent feature(s) for clip_bounds_3979={clip_bounds_3979}"
    )
    return feature_l


def _02_read_dem_non_windowed(
    asset_hrefs: list[str],
    first_crs,
    clip_bounds: tuple[float, float, float, float],
    dst_nodata: float,
    out_path: Path,
    base_profile: dict[str, object],
    use_cache: bool,
    request_token: str,
    logger=None,
) -> dict[str, object]:
    """Read/write DEM using one full-scene merge in memory with per-tile GeoTIFF caching."""
    log = logger or logging.getLogger(__name__)
    tile_shape = (int(base_profile["height"]), int(base_profile["width"]))
    tile_cache_key = _build_tile_cache_key(
        request_token=request_token,
        tile_bounds=clip_bounds,
        tile_shape=tile_shape,
        source_crs=first_crs,
        asset_hrefs=asset_hrefs,
    )
    write_meta_d = {"any_valid": False, "merged_shape": tile_shape}

    def _write_non_windowed_dem(write_fp: Path):
        """Merge one DEM tile from all sources and write to write_fp."""
        src_meta_l = _open_source_meta_list(asset_hrefs, first_crs)
        src_ds_l = [src_meta["ds"] for src_meta in src_meta_l]

        try:
            # Merge once over the target clip bounds.
            merged_data, out_transform = merge(
                src_ds_l,
                bounds=clip_bounds,
                indexes=1,
                nodata=dst_nodata,
                dtype="float32",
                method="last",
            )
        finally:
            _close_source_meta_list(src_meta_l)

        merged = merged_data[0].astype(np.float32, copy=False)
        valid_mask = _build_valid_mask(merged, dst_nodata)
        merged_to_write = _finalize_float32_tile(merged, valid_mask, dst_nodata)
        profile = base_profile.copy()
        profile["transform"] = out_transform
        profile["height"] = int(merged_to_write.shape[0])
        profile["width"] = int(merged_to_write.shape[1])
        with rasterio.open(write_fp, "w", **profile) as dst_ds:
            dst_ds.write(merged_to_write, 1)
        write_meta_d["any_valid"] = bool(valid_mask.any())
        write_meta_d["merged_shape"] = merged_to_write.shape
        log.debug(f"non-windowed merge complete: valid_pixels={int(valid_mask.sum()):,}")

    dem_fp, cache_hit = _materialize_tif_with_cache(
        out_fp=out_path,
        use_cache=bool(use_cache),
        cache_key=tile_cache_key,
        writer=_write_non_windowed_dem,
        logger=log,
    )
    if cache_hit:
        write_meta_d["any_valid"] = _raster_has_any_valid_pixels(dem_fp, dst_nodata)
        with rasterio.open(dem_fp) as ds:
            write_meta_d["merged_shape"] = (int(ds.height), int(ds.width))

    return {
        "dem_fp": dem_fp,
        "any_valid": bool(write_meta_d["any_valid"]),
        "missing_window_count": 0,
        "diag": f"mode=non_windowed, merged_shape={write_meta_d['merged_shape']}, cache_hit={int(cache_hit)}",
    }


def _03_read_dem_windowed_tiles_to_vrt(
    asset_hrefs: list[str],
    first_crs,
    clip_bounds: tuple[float, float, float, float],
    source_res: tuple[float, float],
    depth_transform,
    depth_crs,
    depth_width: int,
    depth_height: int,
    dst_nodata: float,
    fetch_window_size: int,
    out_path: Path,
    use_cache: bool,
    request_token: str,
    use_project_extent_filter: bool,
    project_extent_url: str,
    project_extent_timeout_s: float,
    tqdm_disable: bool = False,
    logger=None,
) -> dict[str, object]:
    """Read DEM windows into per-tile GeoTIFFs and assemble one VRT mosaic."""
    log = logger or logging.getLogger(__name__)
    log.debug(f"_03_read_dem_windowed_tiles_to_vrt start: assets={len(asset_hrefs)}, out_path={out_path}")
    # Normalize and validate the requested fetch tile size.
    fetch_window_size = int(fetch_window_size)
    assert fetch_window_size > 0, f"fetch_window_size must be > 0; got {fetch_window_size}"
    # Derive static grid dimensions and validate source metadata.
    depth_width = int(depth_width)
    depth_height = int(depth_height)
    assert depth_width > 0 and depth_height > 0, f"invalid low-res raster shape: {(depth_height, depth_width)}"
    source_res_x, source_res_y = source_res
    assert source_res_x > 0 and source_res_y > 0, f"invalid source resolution: {source_res}"
    assert project_extent_timeout_s > 0, f"project_extent_timeout_s must be > 0, got {project_extent_timeout_s}"
    # Build the full fetch-tile grid once in fetch-native CRS.
    tile_grid_gdf = _build_fetch_tile_grid_gdf(
        depth_width=depth_width,
        depth_height=depth_height,
        fetch_window_size=fetch_window_size,
        depth_transform=depth_transform,
        depth_crs=depth_crs,
    )
    tile_count = int(len(tile_grid_gdf))
    assert tile_count > 0, "tile grid builder returned no windows"
    tile_rows = int((depth_height + fetch_window_size - 1) // fetch_window_size)
    tile_cols = int((depth_width + fetch_window_size - 1) // fetch_window_size)
    full_tile_count = int(
        ((tile_grid_gdf["lowres_height"] == fetch_window_size) & (tile_grid_gdf["lowres_width"] == fetch_window_size)).sum()
    )
    edge_tile_count = int(tile_count - full_tile_count)
    log.info(
        f"window tiling plan: lowres_window_shape={fetch_window_size:,}x{fetch_window_size:,}, "
        f"lowres_grid_rows={tile_rows:,}, lowres_grid_cols={tile_cols:,}, tiles_total={tile_count:,}, "
        f"full_tiles={full_tile_count:,}, edge_tiles={edge_tile_count:,}"
    )
    tile_grid_gdf = tile_grid_gdf.copy()
    tile_grid_gdf["intersects_project"] = True
    if bool(use_project_extent_filter):
        clip_bounds_3979 = (
            clip_bounds
            if str(first_crs) == "EPSG:3979"
            else transform_bounds(first_crs, "EPSG:3979", *clip_bounds, densify_pts=DEFAULT_BOUNDS_DENSIFY_PTS)
        )
        project_extent_gdf = _build_project_extent_gdf(
            clip_bounds_3979,
            project_extent_url=project_extent_url,
            timeout_s=project_extent_timeout_s,
            logger=log,
        )
        project_extent_for_tiles_gdf = (
            project_extent_gdf
            if str(depth_crs) == "EPSG:3979"
            else project_extent_gdf.to_crs(depth_crs)
        )
        tile_grid_gdf["intersects_project"] = _resolve_fetch_tile_selection_mask(
            tile_grid_gdf,
            project_extent_for_tiles_gdf,
        )
        if not bool(tile_grid_gdf["intersects_project"].any()):
            raise RuntimeError(f"no fetch tiles intersect HRDEM project extent polygons for bounds={clip_bounds_3979}")
        skipped_tile_count = int(tile_count - int(tile_grid_gdf["intersects_project"].sum()))
        if skipped_tile_count > 0:
            log.warning(
                f"project extent polygon prefilter will skip {skipped_tile_count:,}/{tile_count:,} fetch tile(s)"
            )
    tile_grid_iter_gdf = tile_grid_gdf[tile_grid_gdf["intersects_project"]].copy()
    iter_tile_count = int(len(tile_grid_iter_gdf))
    assert iter_tile_count > 0, "project extent filtering removed all fetch tiles"

    # Track run diagnostics and aggregate counters in one metadata dictionary.
    work_nodata = np.float32(DEFAULT_WORK_NODATA)
    diag_d = {"any_valid": False, "missing_window_count": 0, "tiles_with_assets_count": 0, "tiles_with_valid_count": 0}
    tile_meta_l = []
    # Recreate tile output directory to avoid stale files from previous runs.
    tile_dir = out_path.parent / f"{out_path.stem}__fetch_tiles"
    if tile_dir.exists():
        shutil.rmtree(tile_dir)
    tile_dir.mkdir(parents=True, exist_ok=True)
    # Keep the VRT output path in sync with the requested output stem.
    vrt_fp = out_path.with_suffix(".vrt")
    if vrt_fp.exists():
        vrt_fp.unlink()
    log.debug(f"windowed outputs prepared: tile_dir={tile_dir}, vrt_fp={vrt_fp}")

    # Open each source once and validate CRS consistency up front.
    src_meta_l = _open_source_meta_list(asset_hrefs, first_crs)
    assert src_meta_l, "windowed fetch requires at least one opened source dataset"
    log.debug(f"opened source datasets for windowed fetch: count={len(src_meta_l)}")

    try:
        # Iterate output windows, sample overlapping assets, and write one GeoTIFF per window.
        with tqdm(
            total=iter_tile_count,
            desc="HRDEM window fetch",
            unit="tile",
            disable=bool(tqdm_disable),
        ) as progress:
            for i, tile_row in enumerate(tile_grid_iter_gdf.itertuples(index=False), start=1):
                # Emit periodic progress diagnostics for long tiled fetch runs.
                if i == 1 or i % 100 == 0 or i == iter_tile_count:
                    log.debug(
                        f"window progress {i:,}/{iter_tile_count:,}: lowres_row_off={int(tile_row.row_off)}, "
                        f"lowres_col_off={int(tile_row.col_off)}, lowres_height={int(tile_row.lowres_height)}, "
                        f"lowres_width={int(tile_row.lowres_width)}"
                    )
                # Validate expected fixed window shape away from right/bottom edges.
                is_bottom_edge = int(tile_row.row_off) + int(tile_row.lowres_height) >= depth_height
                is_right_edge = int(tile_row.col_off) + int(tile_row.lowres_width) >= depth_width
                if not is_bottom_edge:
                    assert int(tile_row.lowres_height) == fetch_window_size, (
                        f"non-edge window height must equal fetch_window_size={fetch_window_size}, "
                        f"got {int(tile_row.lowres_height)} at lowres_row_off={int(tile_row.row_off)}"
                    )
                if not is_right_edge:
                    assert int(tile_row.lowres_width) == fetch_window_size, (
                        f"non-edge window width must equal fetch_window_size={fetch_window_size}, "
                        f"got {int(tile_row.lowres_width)} at lowres_col_off={int(tile_row.col_off)}"
                    )
                win_bounds = tuple(float(v) for v in tile_row.win_bounds)
                # Convert this tile bounds from depth CRS to source CRS for raster IO.
                fetch_bounds = transform_bounds(
                    depth_crs,
                    first_crs,
                    *win_bounds,
                    densify_pts=DEFAULT_BOUNDS_DENSIFY_PTS,
                )
                # Derive integer output raster shape in source pixels.
                win_width = max(1, int(np.ceil((fetch_bounds[2] - fetch_bounds[0]) / source_res_x)))
                win_height = max(1, int(np.ceil((fetch_bounds[3] - fetch_bounds[1]) / source_res_y)))
                # Build output transform for the fetched source-resolution tile.
                tile_transform = from_bounds(*fetch_bounds, win_width, win_height)

                # Select only source assets that intersect this window's bounds.
                window_src_meta_l = [
                    meta_d
                    for meta_d in src_meta_l
                    if _bounds_intersect(fetch_bounds, meta_d["bounds"])
                ]
                if not window_src_meta_l:
                    diag_d["missing_window_count"] += 1
                    if diag_d["missing_window_count"] <= 5:
                        log.warning(f"no HRDEM assets found for fetch window bounds={win_bounds}; writing nodata")
                else:
                    diag_d["tiles_with_assets_count"] += 1

                # Write this fetched tile to disk with a tile-specific geotransform.
                tile_fp = tile_dir / f"tile_r{int(tile_row.row_off):07d}_c{int(tile_row.col_off):07d}.tif"
                tile_profile = {
                    "driver": "GTiff",
                    "height": win_height,
                    "width": win_width,
                    "count": 1,
                    "dtype": "float32",
                    "crs": first_crs,
                    "transform": tile_transform,
                    "nodata": dst_nodata,
                }
                tile_profile.update(_resolve_gtiff_write_options(win_width, win_height))
                # Key cache by request + exact source-CRS fetch footprint + output shape.
                tile_cache_key = _build_tile_cache_key(
                    request_token=request_token,
                    tile_bounds=fetch_bounds,
                    tile_shape=(win_height, win_width),
                    source_crs=first_crs,
                    asset_hrefs=asset_hrefs,
                )
                # Mutable status carrier used by writer callback to report valid-pixel presence.
                tile_status_d = {"has_valid": False}
                # Bind per-tile arguments once; cache wrapper only passes the destination filepath.
                tile_writer = partial(
                    _write_window_tile,
                    tile_profile=tile_profile,
                    window_src_meta_l=window_src_meta_l,
                    fetch_bounds=fetch_bounds,
                    win_height=win_height,
                    win_width=win_width,
                    dst_nodata=dst_nodata,
                    work_nodata=work_nodata,
                    tile_status_d=tile_status_d,
                )

                # Materialize tile either from cache or by invoking the bound tile writer.
                tile_written_fp, cache_hit = _materialize_tif_with_cache(
                    out_fp=tile_fp,
                    use_cache=bool(use_cache),
                    cache_key=tile_cache_key,
                    writer=tile_writer,
                    logger=log,
                )
                # On cache miss the writer sets this flag; on hit we recompute from the cached raster.
                tile_has_valid = bool(tile_status_d["has_valid"])
                if cache_hit:
                    # Recompute validity from cached tile because writer callback was skipped.
                    tile_has_valid = _raster_has_any_valid_pixels(tile_written_fp, dst_nodata)
                if tile_has_valid:
                    # Track run-level signal for "at least one valid DEM sample was fetched".
                    diag_d["any_valid"] = True
                    diag_d["tiles_with_valid_count"] += 1
                tile_meta_l.append(
                    {
                        "fp": tile_written_fp,
                        "row_off": int(tile_row.row_off),
                        "col_off": int(tile_row.col_off),
                        "height": win_height,
                        "width": win_width,
                    }
                )
                progress.update(1)
    finally:
        # Always close source datasets even if a window read fails.
        _close_source_meta_list(src_meta_l)
    log.debug(f"windowed tile writing complete: tiles_written={len(tile_meta_l)}")

    # Build one mosaic VRT over all tile rasters using GDAL Python bindings.
    tile_fp_l = [str(meta_d["fp"]) for meta_d in tile_meta_l]
    assert tile_fp_l, "windowed fetch produced no tiles for VRT build"
    # Use highest source resolution and explicit nodata semantics in VRT assembly.

    vrt_options = gdal.BuildVRTOptions(
        resolution=DEFAULT_VRT_RESOLUTION,
        srcNodata=float(dst_nodata),
        VRTNodata=float(dst_nodata),
    )
    vrt_ds = gdal.BuildVRT(str(vrt_fp), tile_fp_l, options=vrt_options)
    assert vrt_ds is not None, f"gdal.BuildVRT returned None for {vrt_fp}"
    vrt_ds = None
    log.debug("built VRT with GDAL Python bindings")
    # Return diagnostics for common wrap-up handling.
    log.debug(
        f"windowed fetch diagnostics: total_tiles={tile_count:,}, "
        f"tiles_with_assets={diag_d['tiles_with_assets_count']:,}, tiles_with_valid_pixels={diag_d['tiles_with_valid_count']:,}, "
        f"missing_tiles={diag_d['missing_window_count']:,}"
    )
    return {
        "dem_fp": vrt_fp,
        "any_valid": diag_d["any_valid"],
        "missing_window_count": diag_d["missing_window_count"],
        "diag": f"mode=windowed_vrt, tiles={tile_count}",
    }


def write_dem_from_asset_hrefs(
    depth_lr_fp: str | Path,
    asset_hrefs: list[str],
    output_fp: str | Path,
    request_token: str | None = None,
    use_cache: bool = False,
    logger=None,
    fetch_window_size: int | None = None,
    use_project_extent_filter: bool = False,
    project_extent_url: str = PROJECT_EXTENT_URL,
    project_extent_timeout_s: float = DEFAULT_PROJECT_EXTENT_TIMEOUT_S,
    tqdm_disable: bool = False,
) -> Path:
    """Build one clipped DEM from preselected assets, with per-GeoTIFF tile caching."""
    log = logger or logging.getLogger(__name__)
    # Resolve query geometry once, then derive source-grid fetch geometry.
    depth_query = _resolve_depth_query_geometry(depth_lr_fp)
    if request_token is None:
        request_token = _build_request_token(
            depth_bounds=depth_query["depth_bounds"],
            stac_url="direct_assets",
            collection="direct_assets",
            asset_key="direct_assets",
            tiling_enabled=bool(fetch_window_size is not None),
            fetch_window_size=0 if fetch_window_size is None else int(fetch_window_size),
            use_project_extent_filter=bool(use_project_extent_filter),
        )
    assert asset_hrefs, "asset_hrefs must not be empty"
    log.debug(
        f"write_dem_from_asset_hrefs inputs: depth_lr_fp={depth_query['depth_fp']}, "
        f"depth_crs={depth_query['depth_crs']}, depth_bounds={depth_query['depth_bounds']}, "
        f"depth_shape={(int(depth_query['depth_height']), int(depth_query['depth_width']))}, "
        f"asset_count={len(asset_hrefs)}, request_token={request_token}, use_cache={use_cache}, "
        f"use_project_extent_filter={use_project_extent_filter}"
    )

    out_path = Path(output_fp).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    geom = _estimate_fetch_geometry(depth_query["depth_crs"], depth_query["depth_bounds"], asset_hrefs[0])
    first_crs = geom["source_crs"]
    source_nodata = geom["source_nodata"]
    clip_bounds = geom["clip_bounds"]
    est_width = int(geom["width"])
    est_height = int(geom["height"])
    est_pixels = int(geom["pixels"])
    est_float32_gb = float(geom["float32_gib"])
    out_transform = geom["out_transform"]
    dst_nodata = float(source_nodata) if source_nodata is not None else -9999.0
    log.debug(f"reference asset CRS={first_crs}, clip_bounds={clip_bounds}")
    log.info(
        f"raw fetch request grid: width={est_width:,}, height={est_height:,}, "
        f"pixels={est_pixels:,}, float32_estimate={est_float32_gb:.2f} GiB"
    )

    base_profile = {
        "driver": "GTiff",
        "height": int(est_height),
        "width": int(est_width),
        "count": 1,
        "dtype": "float32",
        "crs": first_crs,
        "transform": out_transform,
        "nodata": dst_nodata,
    }
    base_profile.update(_resolve_gtiff_write_options(int(est_width), int(est_height)))

    # Dispatch read mode: one in-memory merge or tiled-to-disk + VRT.
    if fetch_window_size is None:
        result_d = _02_read_dem_non_windowed(
            asset_hrefs,
            first_crs=first_crs,
            clip_bounds=clip_bounds,
            dst_nodata=dst_nodata,
            out_path=out_path,
            base_profile=base_profile,
            use_cache=bool(use_cache),
            request_token=request_token,
            logger=log,
        )
    else:
        result_d = _03_read_dem_windowed_tiles_to_vrt(
            asset_hrefs,
            first_crs=first_crs,
            clip_bounds=clip_bounds,
            source_res=geom["source_res"],
            depth_transform=depth_query["depth_transform"],
            depth_crs=depth_query["depth_crs"],
            depth_width=int(depth_query["depth_width"]),
            depth_height=int(depth_query["depth_height"]),
            dst_nodata=dst_nodata,
            fetch_window_size=int(fetch_window_size),
            out_path=out_path,
            use_cache=bool(use_cache),
            request_token=request_token,
            use_project_extent_filter=bool(use_project_extent_filter),
            project_extent_url=project_extent_url,
            project_extent_timeout_s=float(project_extent_timeout_s),
            tqdm_disable=bool(tqdm_disable),
            logger=log,
        )

    # Common wrap-up validation and logging for both read modes.
    if not bool(result_d["any_valid"]):
        raise RuntimeError(
            f"no valid DEM pixels found across {len(asset_hrefs)} assets for bounds={depth_query['depth_bounds']}"
        )
    missing_window_count = int(result_d["missing_window_count"])
    if missing_window_count > 0:
        log.warning(f"no HRDEM assets found for {missing_window_count} fetch window(s); wrote nodata in those regions")
    dem_fp = Path(result_d["dem_fp"]).resolve()
    log.info(f"wrote fetched HRDEM tile to\n    {dem_fp}")
    return dem_fp


def main_fetch_hrdem_for_lowres_tile(
    depth_lr_fp: str | Path,
    output_fp: str | Path | None = None,
    logger=None,
    stac_url: str = STAC_URL,
    collection: str = COLLECTION,
    asset_key: str = DEFAULT_ASSET,
    use_cache: bool = True,
    force_tiling: bool = False,
    fetch_window_size: int = DEFAULT_FETCH_WINDOW_SIZE,
    memory_limit_gib: float = DEFAULT_FETCH_MEMORY_LIMIT_GIB,
    stac_query_limit: int = DEFAULT_STAC_QUERY_LIMIT,
    use_project_extent_filter: bool = True,
    project_extent_url: str = PROJECT_EXTENT_URL,
    project_extent_timeout_s: float = DEFAULT_PROJECT_EXTENT_TIMEOUT_S,
    tqdm_disable: bool = False,
) -> DemFetchResult:
    """Fetch one HRDEM tile aligned to a low-res depth raster query footprint."""
    log = logger or logging.getLogger(__name__)
    assert isinstance(use_cache, bool), f"use_cache must be bool, got {type(use_cache)!r}"
    assert isinstance(force_tiling, bool), f"force_tiling must be bool, got {type(force_tiling)!r}"
    assert isinstance(use_project_extent_filter, bool), (
        f"use_project_extent_filter must be bool, got {type(use_project_extent_filter)!r}"
    )
    assert isinstance(tqdm_disable, bool), f"tqdm_disable must be bool, got {type(tqdm_disable)!r}"
    fetch_window_size = int(fetch_window_size)
    assert fetch_window_size > 0, f"fetch_window_size must be > 0; got {fetch_window_size}"
    assert memory_limit_gib > 0, f"memory_limit_gib must be > 0; got {memory_limit_gib}"
    stac_query_limit = int(stac_query_limit)
    assert stac_query_limit > 0, f"stac_query_limit must be > 0; got {stac_query_limit}"
    assert project_extent_timeout_s > 0, f"project_extent_timeout_s must be > 0, got {project_extent_timeout_s}"
    # Resolve low-res query geometry once for both cache keying and STAC search.
    depth_query = _resolve_depth_query_geometry(depth_lr_fp)
    depth_path = depth_query["depth_fp"]
    depth_crs = depth_query["depth_crs"]
    depth_bounds = depth_query["depth_bounds"]
    bbox_4326 = depth_query["bbox_4326"]

    log.info(
        "starting DEM fetch\n"
        f"  source_id={SOURCE_ID}\n"
        f"  stac_url={stac_url}\n"
        f"  collection={collection}\n"
        f"  asset_key={asset_key}\n"
        f"  force_tiling={force_tiling}\n"
        f"  fetch_window_size={fetch_window_size}\n"
        f"  memory_limit_gib={memory_limit_gib:.2f}\n"
        f"  stac_query_limit={stac_query_limit}\n"
        f"  use_project_extent_filter={use_project_extent_filter}\n"
        f"  use_cache={use_cache}\n"
        f"  tqdm_disable={tqdm_disable}\n"
        f"  project_extent_url={project_extent_url}\n"
        f"  depth_lr_fp=\n    {depth_path}"
    )

    # Discover candidate assets from STAC using the query bbox in WGS84.
    # STAC bbox search is intentionally coarse and can over-return assets near edges.
    item_ids, asset_hrefs, _ = _query_hrdem_assets(
        bbox_4326=bbox_4326,
        stac_url=stac_url,
        collection=collection,
        asset_key=asset_key,
        stac_query_limit=stac_query_limit,
    )
    # Build source-CRS fetch geometry and enforce exact raster-bound filtering.
    # Why this second filter is required:
    # 1) BBOX-only discovery can return false positives that intersect the bbox but not the exact footprint.
    # 2) CRS transforms/densification can amplify edge effects, especially for narrow query extents.
    # 3) Dropping non-overlapping assets early avoids unnecessary reads/downloads and nodata-only tiles.
    # 4) This keeps tile diagnostics more stable and deterministic for both local and network runs.
    fetch_geom = _estimate_fetch_geometry(depth_crs, depth_bounds, asset_hrefs[0])
    item_ids, asset_hrefs = _filter_hrdem_assets_to_clip_bounds(
        item_ids,
        asset_hrefs,
        clip_bounds=fetch_geom["clip_bounds"],
        expected_crs=fetch_geom["source_crs"],
        logger=log,
    )
    if not asset_hrefs:
        raise RuntimeError(
            f"HRDEM STAC candidate assets did not intersect the exact query footprint for bounds={depth_bounds}"
        )
    log.info(
        f"found {len(item_ids)} HRDEM item(s) intersecting low-res tile bounds after exact intersection filter"
    )
    est_float32_gib = float(fetch_geom["float32_gib"])
    tiling_enabled = bool(force_tiling or est_float32_gib > float(memory_limit_gib))
    if force_tiling:
        log.warning("forcing tiled fetch via configuration")
    elif tiling_enabled:
        log.warning(
            f"auto-enabling tiled fetch: estimated float32 raster memory {est_float32_gib:.2f} GiB "
            f"exceeds limit {memory_limit_gib:.2f} GiB"
        )
    else:
        log.debug(
            f"estimated float32 raster memory {est_float32_gib:.2f} GiB within limit {memory_limit_gib:.2f} GiB; "
            "using non-tiled fetch"
        )

    # Build one deterministic token used for both default temp output naming and tile-cache key namespacing.
    # Including tiling/project-extent mode prevents collisions between different fetch strategies for the same bounds.
    request_token = _build_request_token(
        depth_bounds=depth_bounds,
        stac_url=stac_url,
        collection=collection,
        asset_key=asset_key,
        tiling_enabled=tiling_enabled,
        fetch_window_size=fetch_window_size,
        use_project_extent_filter=bool(tiling_enabled and use_project_extent_filter),
    )
    if output_fp is None:
        suffix = ".vrt" if tiling_enabled else ".tif"
        out_path = _resolve_default_output_path(request_token, suffix=suffix)
    else:
        out_path = Path(output_fp).expanduser().resolve()
        if tiling_enabled and out_path.suffix.lower() != ".vrt":
            out_path = out_path.with_suffix(".vrt")
    # Cache lives at the GeoTIFF tile level only; VRT is an assembly artifact rebuilt per request.
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written_fp = write_dem_from_asset_hrefs(
        depth_lr_fp=depth_path,
        asset_hrefs=asset_hrefs,
        output_fp=out_path,
        request_token=request_token,
        use_cache=bool(use_cache),
        logger=log,
        fetch_window_size=fetch_window_size if tiling_enabled else None,
        use_project_extent_filter=bool(tiling_enabled and use_project_extent_filter),
        project_extent_url=project_extent_url,
        project_extent_timeout_s=float(project_extent_timeout_s),
        tqdm_disable=bool(tqdm_disable),
    )
    return DemFetchResult(
        source_id=SOURCE_ID,
        dem_fp=written_fp,
        stac_url=stac_url,
        collection=collection,
        asset_key=asset_key,
        item_ids=item_ids,
    )
