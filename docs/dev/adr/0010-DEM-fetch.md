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
- add optional `--fetch-out`   flag to specify output path for fetched HRDEM tile. If not provided, the fetched tile will live in the temp directory (e.g. using `tempfile` module) NOT the cahce. should provide some lazy caching so if the same tile is requested in the same fetch session, it doesn't re-fetch from the source. 

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
- if intersection found, fetch the corresponding HRDEM tile(s)
- let gdal/backends handle multi-processing/threading for now (keep our implementation simple). 

see `dev/proof_of_concepts/hrdem_fetch.ipynb`
 


#### post and pre processing changes
- We want to avoid inheriting downstream defaults from HRDEM, but minimize refactoring... so anything we expect to be inherited from the DEM should be set from lores depth onto the HRDEM right after fetch (before pre-processing). including at leastL
    - reprojected to match the low-res depth raster crs and bbox and nodata values. 
- then the `docs/dev/adr/0009-preproccessing.md` pre-processing steps and checks are applied to the fetched HRDEM tile  


 