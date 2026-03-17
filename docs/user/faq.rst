FAQ
===

What is the maximum resolution `tohr` can handle?
-------------------------------------------------
In general, `tohr` outputs a hi-res depth grid at the same resolution as the provided hi-res DEM.
How that resolution enhancement is achieved varies by model:
- `ResUNet_16x_DEM` provides a 16x enhancement with the inference model weights. Any additional resampling needed is achieved with basic bilinear resampling.


So, there is no hard-coded *maximum* resolution.
In theory, `tohr` will resample to any resolution you can find a DEM at.
However, in standard flood hazard contexts, resolutions finer than 1m are rarely useful.
