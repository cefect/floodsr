"""Common contracts for DEM source backends."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DemFetchResult:
    """Structured output for one DEM fetch operation."""

    source_id: str
    dem_fp: Path
    stac_url: str
    collection: str
    asset_key: str
    item_ids: list[str]
