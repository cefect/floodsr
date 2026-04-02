# CostGrow Status Review

Date: 2026-04-01
Branch at review start: `dev_cg`
Review branch: `feat/cg-03-wg-costgrow-review`
Refreshed after pull from `origin/dev_cg` through `ac5dff2` (`tutorial 3 polishing (#68)`)

## Purpose

Capture where the CostGrow work stands relative to `flood-plan.md` after subsequent project changes, and record the main follow-up items for Windows validation.

## Current Status

### Landed

- `CostGrow_Terrain` exists as a built-in model worker in `floodsr/models/CostGrow_Terrain.py`.
- The model registry supports built-in workers with no downloaded artifact.
- CLI model resolution supports built-in workers and `floodsr models list` shows `CostGrow_Terrain` as built-in.
- `floodsr doctor` reports PCRaster diagnostics without crashing when PCRaster is missing.
- CostGrow unit, CLI, integration, and regression hooks exist in `tests/`.
- Regression expectations for CostGrow are recorded in the existing `case_spec.json` flow.
- `README.md` now documents PCRaster as part of the extended install path.
- The deploy lockfile includes `pcraster`.
- Windows validation in a local `floodsr-gdal` conda env now confirms the CostGrow runtime path works end-to-end on at least one committed case.

### Partially Landed

- Platform preprocessing can switch to windowed materialization for large rasters before CostGrow runs.
- PCRaster runtime gating is implemented with a lazy import and a helpful error message.
- Install guidance is only partially aligned:
  - `README.md` mentions `gdal pcraster` for the extended path.
  - The base conda environment spec and install-proof workflow still lag behind that documented requirement.

### Still Missing Or Not Aligned With The Plan

- CostGrow does not yet have a true model-phase large-raster strategy. The worker still reads prepared rasters into memory and runs as a whole-scene global solve.
- The extended install proof does not currently validate a CostGrow-capable environment end-to-end. It still creates the extended env with `gdal` but not `pcraster`, and the workflow assertions only check GDAL fields.
- The base conda environment file does not list PCRaster even though the deploy lockfile and README now expect it.
- User-facing docs are stale in places:
  - `docs/user/user_guide.rst` still says CostGrow is not implemented.
  - `docs/user/getting_started.rst` still says ResUNet is the only available model.
  - `docs/user/tutorials.rst` does not include the planned comparison tutorial.
- Fresh Windows validation on Anaconda base confirms the CLI works from that environment, but the available base env is not CostGrow-capable yet:
  - `floodsr models list` works and shows `CostGrow_Terrain` as built-in.
  - `floodsr doctor --json` reports `pcraster.installed=false`, `pcraster.spreadzone_available=false`, `gdal.python_bindings_installed=false`, and `gdal.vrt_enabled=false`.
  - A CostGrow CLI smoke attempt fails with the expected helpful PCRaster error rather than crashing unexpectedly.

## Regression Snapshot

The branch is functionally integrated enough that the CostGrow path is represented in the regression suite. The committed expectations currently show:

- `2407_FHIMP_tile`: CostGrow runs and has finite regression targets, but performs materially worse than ResUNet on the recorded metrics.
- `rss_dudelange_A`: CostGrow regression targets are much weaker than ResUNet.
- `rss_mersch_A`: CostGrow regression targets are much weaker than ResUNet.

This means the work is beyond scaffold status, but still needs quality evaluation before positioning CostGrow as a broadly competitive alternative.

## Local Validation Performed During Review

Historical validation on the original review environment:

- `python -m floodsr.cli models list`
- `python -m floodsr.cli doctor`
- `python -m pytest -m fast tests/test_model_registry.py tests/test_cli_models.py tests/test_cli_tohr.py tests/test_costgrow_unit.py tests/test_costgrow_integration.py tests/test_engine_contracts.py -q`

Result:

- 54 passed
- 1 skipped
- 1 failed

The single failure was the manifest HTTP link resolution test due local DNS/network failure, not a CostGrow-specific regression.

Refresh validation after the latest pull on this Windows machine:

- Located the current note paths under `E:\floodsr\floodsr`.
- Re-checked the current branch state after merging latest `origin/dev_cg`.
- Re-checked the doc/install files called out in this note.
- Confirmed the plain shell Python was missing required deps and not useful for FloodSR validation.
- Switched to Anaconda base Python at `C:\Users\walte\anaconda3\python.exe`.
- Ran `python -m floodsr.cli models list`.
- Ran `python -m floodsr.cli doctor --json`.
- Ran `python -m floodsr.cli tohr --model-version CostGrow_Terrain --in tests/data/rss_dudelange_A/lowres030.tif --dem tests/data/rss_dudelange_A/hires003_dem.tif --out %TEMP%/costgrow_smoke.tif`.
- Switched to the dedicated `floodsr-gdal` conda env.
- Ran `python -m floodsr.cli doctor --json`.
- Ran `python -m floodsr.cli models list`.
- Ran `python -m floodsr.cli tohr --model-version CostGrow_Terrain --in tests/data/rss_dudelange_A/lowres030.tif --dem tests/data/rss_dudelange_A/hires003_dem.tif --out C:/Users/walte/AppData/Local/Temp/costgrow_smoke.tif`.

Result:

- `models list` succeeded and showed `CostGrow_Terrain` as `(built-in, no download)`.
- `doctor --json` succeeded from Anaconda base and confirmed:
  - `pcraster.installed=false`
  - `pcraster.spreadzone_available=false`
  - `gdal.python_bindings_installed=false`
  - `gdal.gdal_config_installed=false`
  - `gdal.vrt_enabled=false`
- The CostGrow CLI smoke test failed cleanly with: `PCRaster is required for CostGrow_Terrain. Use the extended conda environment with \`pcraster\` installed.`
- `doctor --json` from `floodsr-gdal` succeeded and confirmed:
  - `pcraster.installed=true`
  - `pcraster.spreadzone_available=true`
  - `gdal.python_bindings_installed=true`
  - `gdal.gdal_config_installed=false`
  - `gdal.vrt_enabled=true`
- `models list` from `floodsr-gdal` succeeded and showed `CostGrow_Terrain` as `(built-in, no download)`.
- The CostGrow CLI smoke test succeeded in `floodsr-gdal` on `tests/data/rss_dudelange_A`, writing:
  - `C:\Users\walte\AppData\Local\Temp\costgrow_smoke.tif`
  - size `6,279,503` bytes
- This means the Windows machine now has both a verified base-env failure mode and a verified extended-env success path for CostGrow.

## Windows Validation Focus

When moving this branch to a Windows environment, validate these first:

1. `floodsr doctor`
   - base Anaconda validation already shows `pcraster_installed=False`, `pcraster_spreadzone_available=False`, and no GDAL/VRT capability
   - confirmed in `floodsr-gdal`: `pcraster_installed=True`, `pcraster_spreadzone_available=True`, `gdal.python_bindings_installed=True`, `gdal.vrt_enabled=True`
   - note that `gdal_config_installed` remained `False` in this Windows conda env even though GDAL Python bindings and VRT support worked

2. `floodsr models list`
   - confirmed on Windows Anaconda base and `floodsr-gdal`: `CostGrow_Terrain` appears as `(built-in, no download)`

3. CostGrow CLI smoke test
   - base Anaconda validation already confirms the missing-PCRaster failure path is clear and user-facing
   - confirmed in `floodsr-gdal` on `rss_dudelange_A` with a successful output raster written to `%TEMP%\costgrow_smoke.tif`
   - next useful step is running the committed CostGrow regression entries, not another smoke-only invocation

4. CostGrow regression path
   - run the CostGrow entries from `tests/test_tohr_regression.py` in an environment with PCRaster available

5. Large-raster behavior
   - verify that the current windowed platform-preparation path works
   - treat memory usage carefully because the worker still performs a whole-scene solve after preprocessing

## Recommended Next Work

1. Update stale user docs so the surfaced product state matches the code.
2. Align the base conda environment spec and install-proof workflow with the README by adding and validating PCRaster explicitly.
3. Run the committed CostGrow regression entries on Windows now that the `floodsr-gdal` smoke path is confirmed.
4. Decide whether Windows is supported now or still experimental.
5. Implement or explicitly defer the missing model-phase large-raster strategy for CostGrow.
