# BACKLOG

Open planned work that has been split out from developer ADRs and inline TODOs.

## Model Execution and Tiling

- Implement and test model-phase `windowed + feather` support for `ResUNet_16x_DEM`.
  Source: `docs/dev/adr/models/0001-resunet-16x-dem.md`, `docs/dev/adr/0008-tiling.md`.
- Implement and test model-phase `windowed + feather` support for `CostGrow_Terrain`.
  Source: `docs/dev/adr/models/0002-costgrow-terrain.md`, `docs/dev/adr/0008-tiling.md`.

## ToHR Parameters

- Expose CostGrow-specific knobs through explicit model kwargs or CLI/API parameter handling.
  Known knobs: `dp_coarse_pixel_max`, `decay_frac`, `distance_fill_method`, and `distance_fill_kwargs`.
  Source: issue #46, `docs/dev/adr/models/0002-costgrow-terrain.md`.
- Split shared ToHR args from model-specific worker kwargs before dispatch.
  Source: issue #47, `floodsr/tohr.py`.
- Route CostGrow-only knobs through explicit model kwargs, not shared worker args.
  Source: issue #47, `floodsr/models/CostGrow_Terrain.py`.

## Configuration

- Add user-configurable defaults with explicit precedence:
  `CLI args > environment variables > user config file > package defaults`.
  Source: `docs/dev/adr/0011-parameters.md`.

## Runtime and Platforms

- Add GPU/runtime backend support when it becomes a project goal.
  Candidate engines include GPU ORT, TensorRT, and PyTorch runtime adapters behind `EngineBase`.
  Source: `docs/dev/adr/0000-scope.md`, `docs/dev/adr/0007-platforms.md`, `docs/dev/adr/0015-engine-runtime.md`.
- Decide and implement macOS support.
  Source: `docs/dev/adr/0007-platforms.md`.

## User Interfaces

- Explore a QGIS plugin GUI as a separate project.
  Source: `docs/dev/adr/0000-scope.md`.

## DEM Sources

- Add alternate DEM source backends behind the existing `dem_sources` abstraction.
  Source: `docs/dev/adr/0010-DEM-fetch.md`.

## Buildings

- Implement building-data support using NRCan Automatically Extracted Buildings as the selected initial source.
  Source: `docs/dev/adr/0016-buildings.md`.

## Documentation

- Add an interactive-launch button for rendered tutorial notebooks if/when that becomes a supported docs feature.
  Source: `docs/dev/adr/0018-docs-and-tutorials.md`.
