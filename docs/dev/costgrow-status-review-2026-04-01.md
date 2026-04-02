# CostGrow Status Review

Date: 2026-04-01
Branch at review start: `dev_cg`
Review branch: `codex/costgrow-review-windows`

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

### Partially Landed

- Platform preprocessing can switch to windowed materialization for large rasters before CostGrow runs.
- PCRaster runtime gating is implemented with a lazy import and a helpful error message.
- Docs mention PCRaster as a requirement for CostGrow in installation guidance.

### Still Missing Or Not Aligned With The Plan

- CostGrow does not yet have a true model-phase large-raster strategy. The worker still reads prepared rasters into memory and runs as a whole-scene global solve.
- The extended install proof does not currently validate a CostGrow-capable environment end-to-end. It creates the extended env with GDAL, but not PCRaster, and the workflow assertions do not check PCRaster fields.
- User-facing docs are stale in places:
  - `docs/user/user_guide.rst` still says CostGrow is not implemented.
  - `docs/user/getting_started.rst` still says ResUNet is the only available model.
  - `docs/user/tutorials.rst` does not include the planned comparison tutorial.
- The base conda environment file does not list PCRaster even though the deploy lockfile and container path now expect it.

## Regression Snapshot

The branch is functionally integrated enough that the CostGrow path is represented in the regression suite. The committed expectations currently show:

- `2407_FHIMP_tile`: CostGrow runs and has finite regression targets, but performs materially worse than ResUNet on the recorded metrics.
- `rss_dudelange_A`: CostGrow regression targets are much weaker than ResUNet.
- `rss_mersch_A`: CostGrow regression targets are much weaker than ResUNet.

This means the work is beyond scaffold status, but still needs quality evaluation before positioning CostGrow as a broadly competitive alternative.

## Local Validation Performed During Review

- `python -m floodsr.cli models list`
- `python -m floodsr.cli doctor`
- `python -m pytest -m fast tests/test_model_registry.py tests/test_cli_models.py tests/test_cli_tohr.py tests/test_costgrow_unit.py tests/test_costgrow_integration.py tests/test_engine_contracts.py -q`

Result:

- 54 passed
- 1 skipped
- 1 failed

The single failure was the manifest HTTP link resolution test due local DNS/network failure, not a CostGrow-specific regression.

## Windows Validation Focus

When moving this branch to a Windows environment, validate these first:

1. `floodsr doctor`
   - confirm `pcraster_installed=True`
   - confirm `pcraster_spreadzone_available=True`
   - confirm the expected GDAL/VRT capability for the chosen environment

2. `floodsr models list`
   - confirm `CostGrow_Terrain` appears as `(built-in, no download)`

3. CostGrow CLI smoke test
   - run `floodsr tohr --model-version CostGrow_Terrain --in <lowres> --dem <dem> --out <out>`
   - use one of the committed regression cases first

4. CostGrow regression path
   - run the CostGrow entries from `tests/test_tohr_regression.py` in an environment with PCRaster available

5. Large-raster behavior
   - verify that the current windowed platform-preparation path works
   - treat memory usage carefully because the worker still performs a whole-scene solve after preprocessing

## Recommended Next Work

1. Update stale user docs so the surfaced product state matches the code.
2. Extend install proof to create and validate a PCRaster-capable extended environment.
3. Decide whether Windows is a supported CostGrow target now or still experimental.
4. Implement or explicitly defer the missing model-phase large-raster strategy for CostGrow.
