CLI Reference
=============

Auto-generated from live command help output.

Main Command
------------

.. code-block:: text

   usage: floodsr [-h] [-v] [-q] [--log-level {DEBUG,INFO,WARNING,ERROR}]
                  {models,tohr,doctor} ...
   
   Run FloodSR model, cache, and runtime utility commands.
   
   positional arguments:
     {models,tohr,doctor}
       models              List manifest models or fetch cached model weights.
       tohr                Run one super-resolution pass for a low-res depth
                           raster.
       doctor              Report runtime dependency and provider diagnostics.
   
   options:
     -h, --help            show this help message and exit
     -v, --verbose         Increase logging verbosity (repeatable).
     -q, --quiet           Decrease logging verbosity (repeatable).
     --log-level {DEBUG,INFO,WARNING,ERROR}
                           Explicit log level override.

tohr
----

.. code-block:: text

   usage: floodsr tohr [-h] [--machine-json MACHINE_JSON] --in IN_FP
                       (--dem DEM | -f) [--fetch-out FETCH_OUT]
                       [--fetch-force-tiling] [--out OUT]
                       [--model-version MODEL_VERSION] [--model-path MODEL_PATH]
                       [--manifest MANIFEST] [--cache-dir CACHE_DIR]
                       [--backend {http,file}] [--force] [--max-depth MAX_DEPTH]
                       [--min-depth-threshold MIN_DEPTH_THRESHOLD]
                       [--dem-pct-clip DEM_PCT_CLIP]
                       [--window-method {hard,feather}]
                       [--tile-overlap TILE_OVERLAP] [--tile-size TILE_SIZE]
                       [--crs-policy {strict,use-dem,use-lores}]
                       [--show-progress | --no-show-progress]
   
   Run one super-resolution pass for a low-res depth raster.
   
   options:
     -h, --help            show this help message and exit
     --machine-json MACHINE_JSON
                           Load `tohr` parameters from JSON; explicit CLI flags
                           still take precedence.
     --in IN_FP            Input low-res depth raster path.
     --dem DEM             Input high-res DEM raster path.
     -f, --fetch-hrdem     Fetch HRDEM for the low-res raster footprint instead
                           of passing `--dem`.
     --fetch-out FETCH_OUT
                           Write a fetched HRDEM raster to this path instead of a
                           temporary location.
     --fetch-force-tiling  Force tiled HRDEM fetch windows instead of relying on
                           automatic tiling.
     --out OUT             Output raster path. Defaults to
                           `./<input_stem>_sr<input_suffix>` in the current
                           working directory.
     --model-version MODEL_VERSION
                           Manifest model version to run or fetch when `--model-
                           path` is not provided.
     --model-path MODEL_PATH
                           Use an explicit local ONNX model file instead of
                           resolving from cache/manifest.
     --manifest MANIFEST   Read model metadata from an alternate `models.json`
                           manifest.
     --cache-dir CACHE_DIR
                           Use an alternate cache directory for resolved model
                           weights.
     --backend {http,file}
                           Override model weight retrieval backend selection.
     --force               Redownload a versioned model when cache resolution
                           would otherwise reuse it.
     --max-depth MAX_DEPTH
                           Maximum depth used when clipping low-res depth values
                           before log1p scaling into the model input range.
                           Values above this threshold are capped during
                           preprocessing and after inverse scaling. The current
                           ResUNet_16x_DEM default resolves to 5.0.
     --min-depth-threshold MIN_DEPTH_THRESHOLD
                           Minimum predicted depth retained in the final raster.
                           Values below this threshold are written as 0.0 after
                           inference. The current ResUNet_16x_DEM default
                           resolves to 0.01.
     --dem-pct-clip DEM_PCT_CLIP
                           Percentile used to cap high DEM values before min-max
                           normalization to [0, 1]. Lower values clip high
                           terrain more aggressively; higher values preserve more
                           of the upper tail. Used when explicit DEM
                           normalization stats are unavailable. The current
                           ResUNet_16x_DEM default resolves to 95.0.
     --window-method {hard,feather}
                           Tile mosaicing method used when stitching model
                           windows. `hard` uses non-overlapping tiles with direct
                           writes; `feather` uses overlapping tiles with weighted
                           blending to reduce seam artifacts. The current default
                           is `feather`.
     --tile-overlap TILE_OVERLAP
                           Feather overlap in low-res pixels. Ignored unless
                           `--window-method=feather`.
     --tile-size TILE_SIZE
                           Override the low-res tile size; must match the model
                           LR input size.
     --crs-policy {strict,use-dem,use-lores}
                           Policy for CRS mismatches between the low-res depth
                           raster and DEM.
     --show-progress, --no-show-progress
                           Show model download, DEM fetch, and tiled runtime
                           progress output.

models
------

.. code-block:: text

   usage: floodsr models [-h] {list,fetch} ...
   
   List manifest models or fetch cached model weights.
   
   positional arguments:
     {list,fetch}
       list        List model versions defined in the manifest.
       fetch       Fetch one manifest model into the local cache.
   
   options:
     -h, --help    show this help message and exit

models list
-----------

.. code-block:: text

   usage: floodsr models list [-h] [--manifest MANIFEST]
   
   List model versions defined in the manifest.
   
   options:
     -h, --help           show this help message and exit
     --manifest MANIFEST  Read models from an alternate `models.json` manifest.

models fetch
------------

.. code-block:: text

   usage: floodsr models fetch [-h] [--manifest MANIFEST] [--cache-dir CACHE_DIR]
                               [--backend {http,file}] [--force]
                               [--show-progress | --no-show-progress]
                               version
   
   Fetch one manifest model into the local cache.
   
   positional arguments:
     version               Model version key to fetch from the manifest.
   
   options:
     -h, --help            show this help message and exit
     --manifest MANIFEST   Read models from an alternate `models.json` manifest.
     --cache-dir CACHE_DIR
                           Store downloaded weights in an alternate cache
                           directory.
     --backend {http,file}
                           Override weight retrieval backend selection.
     --force               Redownload even when a valid cached weight file
                           already exists.
     --show-progress, --no-show-progress
                           Show model download progress output.

doctor
------

.. code-block:: text

   usage: floodsr doctor [-h] [--json]
   
   Report runtime dependency and provider diagnostics.
   
   options:
     -h, --help  show this help message and exit
     --json      Emit machine-readable JSON instead of line-oriented text.
