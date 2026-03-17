# floodsr
[![CI](https://github.com/cefect/floodsr/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/cefect/floodsr/actions/workflows/ci.yml)
[![Release](https://github.com/cefect/floodsr/actions/workflows/release.yml/badge.svg?branch=master)](https://github.com/cefect/floodsr/actions/workflows/release.yml)
[![Documentation Status](https://readthedocs.org/projects/floodsr/badge/?version=latest)](https://floodsr.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

See the [Getting Started](https://floodsr.readthedocs.io/en/latest/getting_started.html) guide for a quick intro to using the CLI.
 
List available model versions:

Fetch a model by version into the default cache:

```bash
floodsr models fetch ResUNet_16x_DEM 
```

Enhance a low-resolution flood hazard raster *to high resolution* (``tohr``), fetching the DEM from the [HRDEM Mosaic](https://open.canada.ca/data/en/dataset/0fe65119-e96e-4a57-8bfe-9d9245fba06b) data source.

```bash
floodsr tohr --in lowres032.tif --fetch-hrdem 
```

Enhance with a local DEM file:

```bash
floodsr tohr --in lowres032.tif --dem hires002_dem.tif  

```

Doctor diagnostics:

```bash
floodsr doctor
```

For more details, see the [User Guide](https://floodsr.readthedocs.io/en/latest/user_guide.html).

  
 
