# CostGrow Dependency Slice: How To Use

This document covers the first CostGrow implementation slice on `feat/cg-01-deps`.

## What was done

This slice does not implement the CostGrow model yet. It prepares the runtime and install surfaces so the later implementation can rely on PCRaster cleanly.

Changes included:

- Added `pcraster` to the source conda environment spec in `container/miniforge/environment.yml`.
- Aligned the miniforge Docker build so the deploy image verifies PCRaster is already present instead of installing it in a separate ad hoc step.
- Added a lazy PCRaster runtime probe in `floodsr/engine/pcraster_check.py`.
- Extended `floodsr doctor` so it reports PCRaster availability alongside ONNX Runtime, rasterio, and GDAL.
- Extended install-proof coverage in `.github/workflows/install-edge.yml` so the extended conda install path explicitly includes PCRaster and verifies it through `doctor --json`.
- Updated install docs to describe PCRaster as part of the extended environment.

## What this enables

After this slice:

- Basic `pip` / `pipx` installs should still work without PCRaster.
- Extended conda installs now explicitly include PCRaster.
- Future CostGrow code can call the lazy runtime guard instead of importing PCRaster at module import time.
- A human can tell from `floodsr doctor` whether the environment is ready for CostGrow work.

## Files changed

- `container/miniforge/environment.yml`
- `container/miniforge/Dockerfile`
- `floodsr/engine/pcraster_check.py`
- `floodsr/engine/__init__.py`
- `floodsr/cli.py`
- `.github/workflows/install-edge.yml`
- `docs/user/installation.rst`
- `README.md`
- `tests/test_cli_models.py`
- `tests/test_engine_contracts.py`

## Human test plan

Use the `deploy` conda environment or the `main` devcontainer for validation. This repo’s normal test/runtime target is not the host `base` environment.

Important distinction:

- the runtime validation steps only require `floodsr`
- the test execution steps require the project test toolchain, including `pytest`

If you create a fresh end-user style environment with only:

```bash
python -m pip install floodsr
```

then `floodsr doctor` should work, but `pytest` may not be installed. That is expected.

### 1. Confirm you are in the right environment

Run:

```bash
pwd
printenv CONDA_DEFAULT_ENV
uname -a
```

Expected:

- you are in the repo root
- the active conda env is `deploy` or another intentional extended test env

### 2. Validate the basic install still works without PCRaster

Use a clean environment that does not preinstall GDAL or PCRaster.

Example:

```bash
python -m pip install floodsr
floodsr doctor
```

Expected:

- install succeeds
- `floodsr doctor` succeeds
- output includes:
  - `pcraster_installed=False`
  - `pcraster_spreadzone_available=False`
- no import-time crash occurs just because PCRaster is missing

### 3. Validate the extended conda install path

Create a clean conda environment using the documented extended path:

```bash
conda create -n floodsr-gdal -c conda-forge python=3.12 gdal pcraster -y
conda activate floodsr-gdal
python -m pip install floodsr
python -c "from pcraster import spreadzone; print('pcraster_ok')"
floodsr doctor
```

Expected:

- conda env creation succeeds
- direct PCRaster import succeeds
- `floodsr doctor` succeeds
- output includes:
  - `gdal_config_installed=True`
  - `gdal_vrt_enabled=True`
  - `pcraster_installed=True`
  - `pcraster_spreadzone_available=True`

### 4. Validate the lazy PCRaster guard behavior

In an environment without PCRaster, run:

```bash
python - <<'PY'
from floodsr.engine.pcraster_check import _check_pcraster
try:
    _check_pcraster()
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
PY
```

Expected:

- raises `ImportError`
- message explains PCRaster is required for `CostGrow_Terrain`
- message points the user toward the extended conda environment

### 5. Validate the automated fast tests for this slice

This step is for contributor validation, not bare end-user runtime validation.

If you are in `floodsr-gdal` and only ran:

```bash
python -m pip install floodsr
```

then `pytest` will not be installed yet. Add the dev/test extras first:

```bash
conda activate floodsr-gdal
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_cli_models.py tests/test_engine_contracts.py -m fast
```

Expected:

- tests pass
- doctor payload tests include PCRaster fields
- engine diagnostics tests include the PCRaster probe shape

Preferred alternative:

```bash
conda activate deploy
python -m pytest -q tests/test_cli_models.py tests/test_engine_contracts.py -m fast
```

### 6. Validate install-edge workflow manually

If you are using the local/self-hosted runner flow for install proof:

```bash
gh workflow run install-edge.yml
```

Or run the same commands locally that the workflow now uses for extended cases:

```bash
eval "$(conda shell.bash hook)"
conda create -n floodsr-gdal -c conda-forge python=3.12 gdal pcraster -y
conda activate floodsr-gdal
python -m pip install floodsr
python -c "from pcraster import spreadzone; print('pcraster_ok')"
floodsr doctor --json
```

Expected:

- extended install cases prove both GDAL and PCRaster availability
- basic cases still show PCRaster absent without failing

## What is not done yet

This slice does not include:

- CostGrow model registration
- built-in model resolution
- CostGrow algorithm implementation
- CLI wiring for `CostGrow_Terrain`
- CostGrow regression tests
- tutorial/model docs

Those land in later PRs.
