.. _user_faq:

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


What `crs-policy` should I use?
-------------------------------
CRS stands for *coordinate reference system*.
A CRS tells raster software how pixel locations map onto real-world coordinates, including what the coordinate units mean and where the raster sits on the earth.
This matters for raster work because overlays, clipping, reprojection, resampling, and pixel alignment all depend on both rasters meaningfully sharing one spatial reference.
If you want a deeper introduction, see the `Rasterio georeferencing guide <https://rasterio.readthedocs.io/en/stable/topics/georeferencing.html>`_.

In general, if your low-resolution depth raster and DEM already share the same projected CRS, keep the default ``strict`` behavior.
If they do not match, use ``use-dem`` when you want the final workflow aligned to the DEM CRS, or ``use-lores`` when you need to preserve the low-resolution raster CRS as the target.
If you want a complete different CRS, you'll need to reproject one of the rasters yourself before running `tohr` (e.g., using `rasterio CLI` or `gdalwarp`), and then you can use ``strict`` since they will now match.
For most flood applications, assuming your CRS is a reasonable projected CRS (e.g., UTM or a national grid), the choice between these two policies will not make a noticeable difference in the final flood depth output.
Generally, CRS selection is dicated by project or client standards. 

- **Does it make any difference?** Yes, because the chosen policy decides which raster is reprojected and therefore which CRS becomes the canonical working grid.
- **Is it desirable if my inputs are in the same CRS?** Yes, that is the simplest and most reliable setup because it avoids an extra reprojection step and gives the user  more control and better error handling.
- **Does it affect accuracy?** That depends on the specific transformation needed: because reprojection and resampling can slightly shift values or smooth data, especially when changing the DEM grid.
- **Computation time?** Yes, forcing a reprojection usually adds some extra preprocessing cost compared with inputs that already match.
- **Which CRS policy is applied by default if I run `tohr` but do not specify?** ``strict``.
