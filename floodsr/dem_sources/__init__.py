"""DEM source backends and registry helpers."""

from floodsr.dem_sources.base import DemFetchResult
from floodsr.dem_sources.catalog import fetch_dem

__all__ = ["DemFetchResult", "fetch_dem"]
