"""HRDEM STAC backend implementation."""

import hashlib
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np

import rasterio
from rasterio.merge import merge
from rasterio.warp import transform_bounds
 

from floodsr.dem_sources.base import DemFetchResult


SOURCE_ID = "hrdem"
STAC_URL = "https://datacube.services.geo.ca/api"
COLLECTION = "hrdem-mosaic-1m"
DEFAULT_ASSET = "dtm"


_SESSION_FETCH_CACHE: dict[str, Path] = {}


def _build_fetch_cache_key(
    depth_crs_repr: str,
    depth_bounds: tuple[float, float, float, float],
    stac_url: str,
    collection: str,
    asset_key: str,
) -> str:
    """Build a stable cache key for one fetch request."""
    bounds_token = ",".join(f"{float(v):.8f}" for v in depth_bounds)
    payload = f"{depth_crs_repr}|{bounds_token}|{stac_url}|{collection}|{asset_key}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _resolve_temp_fetch_path(cache_key: str) -> Path:
    """Resolve a temporary on-disk target path for one cache key."""
    temp_root = Path(tempfile.gettempdir()) / "floodsr" / "hrdem-fetch"
    temp_root.mkdir(parents=True, exist_ok=True)
    return (temp_root / f"{cache_key}.tif").resolve()


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
    assert depth_crs is not None, f"low-res depth CRS is required for STAC query: {depth_path}"

    # Translate low-res bounds to WGS84 for STAC query filtering.
    lowres_bbox_4326 = transform_bounds(
        depth_crs,
        "EPSG:4326",
        *depth_bounds,
        densify_pts=21,
    )
    assert lowres_bbox_4326[0] < lowres_bbox_4326[2], f"invalid transformed bbox x ordering: {lowres_bbox_4326}"
    assert lowres_bbox_4326[1] < lowres_bbox_4326[3], f"invalid transformed bbox y ordering: {lowres_bbox_4326}"
    return {
        "depth_fp": depth_path,
        "depth_crs": depth_crs,
        "depth_bounds": depth_bounds,
        "depth_nodata": depth_nodata,
        "bbox_4326": tuple(float(v) for v in lowres_bbox_4326),
    }


def _query_hrdem_assets(
    bbox_4326: tuple[float, float, float, float],
    stac_url: str,
    collection: str,
    asset_key: str,
) -> tuple[list[str], list[str]]:
    """Query STAC and return intersecting item ids with asset hrefs."""
    from pystac_client import Client

    client = Client.open(stac_url)
    search = client.search(
        collections=[collection],
        bbox=list(bbox_4326),
        limit=200,
    )
    # Materialize the search result once so we can validate coverage and assets.
    items = list(search.items())
    if not items:
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
        raise RuntimeError(
            f"HRDEM STAC returned items but no '{asset_key}' assets for bbox={bbox_4326}"
        )
    return item_ids, asset_hrefs


def write_dem_from_asset_hrefs(
    depth_lr_fp: str | Path,
    asset_hrefs: list[str],
    output_fp: str | Path,
    logger=None,
) -> Path:
    """Build and write one clipped DEM mosaic from asset hrefs without CRS reprojection."""


    log = logger or logging.getLogger(__name__)
    # Read low-res geometry once and reuse it for source-CRS clipping bounds.
    depth_query = _resolve_depth_query_geometry(depth_lr_fp)
    depth_crs = depth_query["depth_crs"]
    depth_bounds = depth_query["depth_bounds"]
    assert asset_hrefs, "asset_hrefs must not be empty"

    out_path = Path(output_fp).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use the first asset as the reference grid and source CRS for output writing.
    with rasterio.open(asset_hrefs[0]) as first_ds:
        first_crs = first_ds.crs
        assert first_crs is not None, f"asset CRS is required: {asset_hrefs[0]}"
        source_nodata = first_ds.nodata
        clip_bounds = transform_bounds(
            depth_crs,
            first_crs,
            *depth_bounds,
            densify_pts=21,
        )

    # Validate transformed query bounds in the source CRS.
    assert clip_bounds[0] < clip_bounds[2], f"invalid transformed clip bounds x ordering: {clip_bounds}"
    assert clip_bounds[1] < clip_bounds[3], f"invalid transformed clip bounds y ordering: {clip_bounds}"
    dst_nodata = float(source_nodata) if source_nodata is not None else -9999.0

    # Open all assets and enforce one shared CRS to avoid implicit reprojection.
    src_ds_l = []
    for href in asset_hrefs:
        src_ds = rasterio.open(href)
        src_crs = src_ds.crs
        assert src_crs is not None, f"asset CRS is required: {href}"
        if src_crs != first_crs:
            src_ds.close()
            for opened_ds in src_ds_l:
                opened_ds.close()
            raise AssertionError(
                f"all HRDEM assets must share one CRS without auto-reprojection: {src_crs} != {first_crs} for {href}"
            )
        src_ds_l.append(src_ds)

    try:
        # Mosaic and clip in source CRS only; this intentionally avoids depth-CRS reprojection.
        merged_data, out_transform = merge(
            src_ds_l,
            bounds=clip_bounds,
            indexes=1,
            nodata=dst_nodata,
            dtype="float32",
            method="last",
        )
    finally:
        for src_ds in src_ds_l:
            src_ds.close()

    merged = merged_data[0].astype(np.float32, copy=False)
    valid_mask = ~np.isnan(merged) if np.isnan(dst_nodata) else ~np.isclose(merged, np.float32(dst_nodata))

    # Fail hard if no source contributes valid pixels to the target footprint.
    if not valid_mask.any():
        raise RuntimeError(f"no valid DEM pixels found across {len(asset_hrefs)} assets for bounds={depth_bounds}")

    merged_to_write = np.where(valid_mask, merged, np.float32(dst_nodata)).astype(np.float32, copy=False)

    # Persist one aligned float32 DEM tile.
    profile = {
        "driver": "GTiff",
        "height": int(merged_to_write.shape[0]),
        "width": int(merged_to_write.shape[1]),
        "count": 1,
        "dtype": "float32",
        "crs": first_crs,
        "transform": out_transform,
        "nodata": dst_nodata,
        "compress": "LZW",
    }
    with rasterio.open(out_path, "w", **profile) as dst_ds:
        dst_ds.write(merged_to_write, 1)

    log.info(f"wrote fetched HRDEM tile to\n    {out_path}")
    return out_path


def fetch_hrdem_for_lowres_tile(
    depth_lr_fp: str | Path,
    output_fp: str | Path | None = None,
    logger=None,
    stac_url: str = STAC_URL,
    collection: str = COLLECTION,
    asset_key: str = DEFAULT_ASSET,
) -> DemFetchResult:
    """Fetch one HRDEM tile aligned to a low-res depth raster query footprint."""
    log = logger or logging.getLogger(__name__)
    # Resolve low-res query geometry once for both cache keying and STAC search.
    depth_query = _resolve_depth_query_geometry(depth_lr_fp)
    depth_path = depth_query["depth_fp"]
    depth_crs = depth_query["depth_crs"]
    depth_bounds = depth_query["depth_bounds"]
    bbox_4326 = depth_query["bbox_4326"]
    depth_crs_repr = depth_crs.to_string() if depth_crs is not None else "unknown"

    log.info(
        "starting DEM fetch\n"
        f"  source_id={SOURCE_ID}\n"
        f"  stac_url={stac_url}\n"
        f"  collection={collection}\n"
        f"  asset_key={asset_key}\n"
        f"  depth_lr_fp=\n    {depth_path}"
    )
    # Build a deterministic request key for in-process cache reuse.
    cache_key = _build_fetch_cache_key(
        depth_crs_repr=depth_crs_repr,
        depth_bounds=depth_bounds,
        stac_url=stac_url,
        collection=collection,
        asset_key=asset_key,
    )

    item_ids: list[str] = []
    cached_fp = _SESSION_FETCH_CACHE.get(cache_key)
    if cached_fp is not None and cached_fp.exists():
        # Serve cache hit immediately, optionally copying to an explicit output target.
        if output_fp is None:
            return DemFetchResult(
                source_id=SOURCE_ID,
                dem_fp=cached_fp,
                stac_url=stac_url,
                collection=collection,
                asset_key=asset_key,
                item_ids=item_ids,
            )
        out_path = Path(output_fp).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path != cached_fp:
            shutil.copy2(cached_fp, out_path)
        return DemFetchResult(
            source_id=SOURCE_ID,
            dem_fp=out_path,
            stac_url=stac_url,
            collection=collection,
            asset_key=asset_key,
            item_ids=item_ids,
        )

    # Cache miss: discover intersecting assets from STAC.
    item_ids, asset_hrefs = _query_hrdem_assets(
        bbox_4326=bbox_4326,
        stac_url=stac_url,
        collection=collection,
        asset_key=asset_key,
    )
    log.info(f"found {len(item_ids)} HRDEM item(s) intersecting low-res tile bounds")

    # Write to temporary cache path unless caller provided a destination.
    target_fp = _resolve_temp_fetch_path(cache_key) if output_fp is None else Path(output_fp).expanduser().resolve()
    written_fp = write_dem_from_asset_hrefs(
        depth_lr_fp=depth_path,
        asset_hrefs=asset_hrefs,
        output_fp=target_fp,
        logger=log,
    )
    _SESSION_FETCH_CACHE[cache_key] = written_fp
    return DemFetchResult(
        source_id=SOURCE_ID,
        dem_fp=written_fp,
        stac_url=stac_url,
        collection=collection,
        asset_key=asset_key,
        item_ids=item_ids,
    )
