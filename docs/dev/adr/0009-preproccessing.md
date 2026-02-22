# ADR 0009: Pre-processing, input validation, and platform-model boundary
To support modularization and multiple inference engines, pre-processing has two stages:
- model-agnostic platform level (see below) to obtain the **platform-model boundary**.
- model-specific input transformations (e.g., log-scaling, normalization, etc.) that are required to meet the **model-engine boundary**. see `0005-model-registry.md` and `0015-engine-runtime.md`. these happen after platform level. 

see `docs/dev/adr/0001-architecture-and-cli.md`

Implementation note:
- platform-level preprocessing is orchestrated by `floodsr tohr` and is not a standalone CLI entrypoint.
- shared preprocessing logic should live in `floodsr/preprocessing.py`.


## platform-model boundary
Requirements:
- Tool should support sloppy data from the user... as much as possible. otherwise people won't use it.
- cleaning and pre-processing should be as shared as possible. i.e., model-specific pre-processing/cleaning should be minimized. 

### contract
the boundary is defined as:
- identical bbox  
- identical crs
- square pixels
- all masked pixels are also nodata (and raster has a defined nodata value)
  - nodata value (e.g., -9999) is the same between dem and depths


## input handling
### inputs that should be rejected:

raise a verbose assertion error telling the user to fix if the following are not met by input rasters:
- identical crs
- projected crs
- nodata != 0 (this is a valid depth)
- everything is invalid/masked
- one invalid data-signal criteria:
    - none (i.e., everything valid)
    - all masked pixels are also nodata
    - some nodata and NO masked
    - some masked and NO nodata

### inputs that are accepted and warned:
raise a warning and correct if the following are not met:
- square pixels: resample to square
- identical bbox: take lores depth bbox as ground truth and crop/resample dem to match
- nodata signal/mask differs between dem and depth: take 'invalid data signal' of DEM as ground truth, and convert to a mask layer.


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


## nodata normalization/handling
Requirements:
- models can differ on how they treat/handle nodata.
  - e.g., ResUNet_16x_DEM expects all real at the 

## pre-processing to obtain the platform-model boundary contract

- This stage is executed as part of `tohr` before model-worker execution.
- Keep this logic shared/model-agnostic in `floodsr/preprocessing.py` where practical.



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
