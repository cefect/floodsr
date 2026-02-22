CLI Reference
=============

Auto-generated from live command help output.

Main Command
------------

.. code-block:: text

   usage: floodsr [-h] [-v] [-q] [--log-level {DEBUG,INFO,WARNING,ERROR}]
                  {models,tohr,doctor} ...
   
   FloodSR command line interface.
   
   positional arguments:
     {models,tohr,doctor}
       models              Model registry commands.
       tohr                Run one raster ToHR pass.
       doctor              Report runtime dependency diagnostics.
   
   options:
     -h, --help            show this help message and exit
     -v, --verbose         Increase logging verbosity (repeatable).
     -q, --quiet           Decrease logging verbosity (repeatable).
     --log-level {DEBUG,INFO,WARNING,ERROR}
                           Explicit log level override.

tohr
----

.. code-block:: text

   usage: floodsr tohr [-h] --in IN_FP (--dem DEM | -f) [--fetch-out FETCH_OUT]
                       [--out OUT] [--model-version MODEL_VERSION]
                       [--model-path MODEL_PATH] [--manifest MANIFEST]
                       [--cache-dir CACHE_DIR] [--backend {http,file}] [--force]
                       [--max-depth MAX_DEPTH] [--dem-pct-clip DEM_PCT_CLIP]
                       [--window-method {hard,feather}]
                       [--tile-overlap TILE_OVERLAP] [--tile-size TILE_SIZE]
   
   options:
     -h, --help            show this help message and exit
     --in IN_FP            Low-res depth raster path.
     --dem DEM             High-res DEM raster path.
     -f, --fetch-hrdem     Fetch HRDEM from STAC using the low-res raster
                           footprint.
     --fetch-out FETCH_OUT
                           Optional output path for fetched HRDEM tile. Defaults
                           to temp directory.
     --out OUT             Output high-res depth raster path. Defaults to
                           ./<input_stem>_sr with input extension
     --model-version MODEL_VERSION
                           Model version key from manifest when --model-path is
                           not provided.
     --model-path MODEL_PATH
                           Explicit local ONNX model path.
     --manifest MANIFEST   Optional path to an alternate models.json manifest.
     --cache-dir CACHE_DIR
                           Optional cache directory for downloaded weights.
     --backend {http,file}
                           Override retrieval backend selection for model fetch.
     --force               Force redownload when fetching a versioned model.
     --max-depth MAX_DEPTH
                           Optional max depth override for log-space scaling.
     --dem-pct-clip DEM_PCT_CLIP
                           Optional DEM percentile clip override when train stats
                           are incomplete.
     --window-method {hard,feather}
                           Tile mosaicing method for ToHR.
     --tile-overlap TILE_OVERLAP
                           Feather overlap in low-res pixels. Ignored unless
                           --window-method=feather.
     --tile-size TILE_SIZE
                           LR tile size override (must match model LR input
                           size).

models
------

.. code-block:: text

   usage: floodsr models [-h] {list,fetch} ...
   
   positional arguments:
     {list,fetch}
       list        List available model versions.
       fetch       Fetch model weights by version.
   
   options:
     -h, --help    show this help message and exit

models list
-----------

.. code-block:: text

   usage: floodsr models list [-h] [--manifest MANIFEST]
   
   options:
     -h, --help           show this help message and exit
     --manifest MANIFEST  Optional path to an alternate models.json manifest.

models fetch
------------

.. code-block:: text

   usage: floodsr models fetch [-h] [--manifest MANIFEST] [--cache-dir CACHE_DIR]
                               [--backend {http,file}] [--force]
                               version
   
   positional arguments:
     version               Model version key from the manifest.
   
   options:
     -h, --help            show this help message and exit
     --manifest MANIFEST   Optional path to an alternate models.json manifest.
     --cache-dir CACHE_DIR
                           Optional cache directory for downloaded weights.
     --backend {http,file}
                           Override retrieval backend selection.
     --force               Force redownload even when a valid cache file exists.

doctor
------

.. code-block:: text

   usage: floodsr doctor [-h]
   
   options:
     -h, --help  show this help message and exit
