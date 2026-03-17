# floodsr
[![CI](https://github.com/cefect/floodsr/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/cefect/floodsr/actions/workflows/ci.yml)
[![Release](https://github.com/cefect/floodsr/actions/workflows/release.yml/badge.svg?branch=master)](https://github.com/cefect/floodsr/actions/workflows/release.yml)
[![Documentation Status](https://readthedocs.org/projects/floodsr/badge/?version=latest)](https://floodsr.readthedocs.io/en/latest/)

Super-Resolution for flood hazard rasters.
Ingests lores water grid and hires DEM and infers a hires water grid using the specified model.

- **Documentation**: https://floodsr.readthedocs.io/en/latest/
- **Contribute**: https://github.com/cefect/floodsr/blob/master/CONTRIBUTING.md

Implemented models (see `floodsr/models.json`):
- **ResUNet_16x_DEM**: 16x DEM-conditioned ResUNet
 

## Installation

see [documentation](https://floodsr.readthedocs.io/en/latest/installation.html) for details.

### basic install

```bash
pipx install floodsr
```
 
### extended install
for handling rasters too large for memory, floodsr requires GDAL backends.
```bash
# advanced install for VRT workflows
conda create -n floodsr-gdal -c conda-forge python=3.12 gdal -y
conda activate floodsr-gdal
python -m pip install floodsr
```
 

## Use

 
List available model versions:

```bash
floodsr models list
```

Fetch a model by version into the default cache:

```bash
floodsr models fetch ResUNet_16x_DEM 
```

Enhance the provided test raster *to high resolution* (``tohr``), fetching the DEM from the [HRDEM Mosaic](https://open.canada.ca/data/en/dataset/0fe65119-e96e-4a57-8bfe-9d9245fba06b) data source:
```bash
floodsr tohr --in tests/data/2407_FHIMP_tile/lowres032.tif 
```

Run one tohr pass from raster inputs:

```bash
# simple tile
floodsr tohr \
  --in tests/data/2407_FHIMP_tile/lowres032.tif \
  --dem tests/data/2407_FHIMP_tile/hires002_dem.tif  

# larger raster w/ windowing and tiling and rescaling
floodsr tohr \
  --in tests/data/rss_mersch_A/lowres030.tif \
  --dem tests/data/rss_mersch_A/hires002_dem.tif \
  --out pred_sr.tif
 
```

Run ToHR with explicit local model path:

```bash
floodsr tohr \
  --in tests/data/2407_FHIMP_tile/lowres032.tif \
  --dem tests/data/2407_FHIMP_tile/hires002_dem.tif \
  --out ./tmp/pred_sr.tif \
  --model-path _inputs/ResUNet_16x_DEM/model_infer.onnx
```



Doctor diagnostics:

```bash
floodsr doctor
```

 
