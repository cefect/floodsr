# floodsr
[![Documentation Status](https://readthedocs.org/projects/floodsr/badge/?version=latest)](https://floodsr.readthedocs.io/en/latest/)

Super-Resolution for flood hazard rasters.
Ingests lores water grid and hires DEM and infers a hires water grid using the specified model.

Documentation: https://floodsr.readthedocs.io/en/latest/

Implemented models (see `floodsr/models.json`):
- **ResUNet_16x_DEM**: 16x DEM-conditioned ResUNet
- **CostGrow** (future)

Implemented backend:
- **ONNX Runtime**


## Installation

```bash
# recommended: isolated CLI install for users
python -m pip install --user pipx
pipx ensurepath
pipx install floodsr
```

TestPyPI install:

```bash
pipx install --index-url https://test.pypi.org/simple/ --pip-args="--extra-index-url https://pypi.org/simple" floodsr
```

Developer install:

```bash
pip install -e ".[dev]"
```


## Use

Current CLI surface includes model registry, `tohr` raster execution, and runtime diagnostics.

List available model versions:

```bash
# dev shortcut for the CLI
alias floodsr='python -m floodsr.cli'

floodsr models list
```

Fetch a model by version into the default cache:

```bash
floodsr models fetch ResUNet_16x_DEM --force
```

tohr using HRDEM as DEM
```bash
floodsr tohr -f --in tests/data/2407_FHIMP_tile/lowres032.tif 
```

Run one ToHR pass from raster inputs:

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

 
