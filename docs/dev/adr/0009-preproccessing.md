# ADR 0009: Pre-processing and input validation


#### geospatial conformity and nodata handling
Users may pass inputs that are slightly misaligned from model expectations, which are:
- identical bbox  
- identical crs
- square pixels
- consitent nodata metaddata and masks
  - nodata != 0 (this is a valid depth)
  - nodata value (e.g., -9999) is the same between dem and depths
  - one “invalid-data signal”. one of the below or throw an error:
    - none (i.e., everything valid)
    - all masked pixels are also nodata
    - some nodata and NO masked
    - some masked and NO nodata


raise a verbose assertion error telling the user to fix if the following are not met by input rasters:
- identical crs
- projected crs
- one invalid data-signal criteria
- nodata ==0

raise a warning and correct if the following are not met:
- square pixels: resample to square
- identical bbox: take lores depth bbox as ground truth and crop/resample dem to match
- nodata signal/mask differs between dem and depth: take 'invalid data signal' of DEM as ground truth, and convert to a mask layer.

#### nodata normalization/handling
read nodata from DEM mask (see above)
set mask to nodata then replace nodata with:
```bash
gdal raster fill-nodata \
  --overwrite \
  --strategy invdist \
  --format GTiff \
  --co TILED=YES \
  --co BIGTIFF=IF_SAFER \
  --smoothing-iterations 0 \
  "$dem_fp" "$out_fp" \
  > "$log_fp" 2>&1
```
do the same for depths (using the DEM mask)
run inference on the filled rasters (ignore mask)
re-apply  mask in post-processing (masked pixels should also be set to nodata using the nodata value from the DEM)


#### resampling and scale
users may pass arbirary hires/lores input rasters, but model is hard-coded (infer `MODEL_SCALE` from train_config.json). 
for input combinations where the INPUT_SCALE!=MODEL_SCALE:
- if `MODEL_SCALE/INPUT_SCALE>=2`: throw NotSupportedError
- else: resample the dem to obtain `MODEL_SCALE` with a warning that this is not tested and may produce suboptimal results.

##### supported DEM resolution
model learns based on a fixed geospatial distancee in the DEM (e.g., 2m).
unclear how other resolutions will behave.

infer the `MODEL_DEM_RES` from `train_config.json`:
- try for a key train_config['MODEL_DEM_RES']
- fallback to infering from train_config['dem_fp'] filename to extract 2 from a  pattern like `workflow_outdir/08_chips032/hires002_dem.npy` (i.e. 002)

if `INPUT_DEM_RES != MODEL_DEM_RES`, throw a warning that this is not tested and may produce suboptimal results, but proceed with inference  

for legacy train configs where `MODEL_DEM_RES` cannot be inferred, fallback to:
- `MODEL_DEM_RES = 2.0`

#### output geospatial assertions
post-inference, assert:
- output bbox == incoming lores depth bbox
- output shape == preprocessed DEM shape

#### validating input value ranges
input depths:
- if max>15: 'warning: input depth values exceed 15. ensure this is a depths not a water surface raster'
- throw an error if not float type
- assertion error if min<0.01
- warning if min<0.1 
- warning if no zeros 
- error if no valid (after nodata/mask harmonization)

input dem:
- warn if > 10pct of pixels ==0
- warn if max> 9000 (unlikely to be a terrestrial DEM)
- warn if min< -100 (unlikely to be a terrestrial DEM)
