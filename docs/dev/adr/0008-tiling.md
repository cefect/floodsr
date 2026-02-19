# strategy for tiling large rasters for inference


## Problem framing and decision variables

Assume:
- Model expects fixed chip size (e.g. 256×256 or 512×512).
- Input raster is arbitrarily large (e.g. 20k×20k) with arbirary blocksize/driver.
- deployed system has enough memory to fit the output tile (but possibly not all the input tiles as well). 


For a fixed-input inference core, the key design variables are:

- **ROI size (chip size)**: the model’s required input spatial size. see `docs/dev/adr/0001-runtime-and-cli-contract.md`
- **Stride / overlap**: stride controls redundancy; overlap is often expressed as a fraction (e.g., 0.25).  
- **Border policy**: padding mode (constant/reflect/replicate), nodata policy, and whether to request “boundless” windows outside the raster extent.  
- **Read granularity**: how you map ROI windows to storage blocks (internal tiles) because raster I/O is performed at block level; poor alignment inflates I/O dramatically.  
<!-- - **Compute scheduling**: task granularity (number of windows per task), GPU batch size, CPU decode parallelism, device transfers, and whether you pipeline (prefetch/decode while GPU runs).   -->
 


## Decision
- Do not materialise the input full raster in memory. Read tiles/windows on demand. Keep ouput in memory until the end, then write out as a single raster (future: consider a memory checkpoint and spill to disk if needed).
- sliding-window inference with overlap and weighted blending 
- Treat parallelism as a first-class design variable
- no additional depencies (for now)
- default to whats shown in `dev/inference.ipynb`
- add cli options for `--window-method` hard and feathered (default)
 