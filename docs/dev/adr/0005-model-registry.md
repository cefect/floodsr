# ADR-0005: Models (Registry, Artifacts, and I/O Contracts)

 
## Context

FloodSR needs a stable way to:
- discover model versions,
- fetch model artifacts with integrity checks,
- define model-specific I/O contracts at the model-engine boundary.

These concerns are model concerns, not engine concerns.

## Decision

- Maintain a `models.json` manifest mapping:
  - `version -> {file_name, url, sha256, description}`
- Store fetched artifacts in cache per `ADR-0012`.
- Enforce `sha256` checksum validation before use.
- Provide CLI commands:
  - `floodsr models list`
  - `floodsr models fetch <version>`
- Define each model type's I/O contract in this ADR (or future model-specific ADR addenda) while keeping runtime backend policy in `ADR-0015`.

## Model Registry Contract

- Manifest file: `floodsr/models.json`
- Required fields per model record:
  - `file_name`
  - `url`
  - `sha256`
- Optional fields:
  - `description`
- Cache path pattern:
  - `<cache_dir>/<model_version>/<file_name>`
- Fetch behavior:
  - reuse cached artifact only when checksum matches
  - otherwise re-download into `*.part` and atomically replace on successful checksum

## Model Types

### 16x DEM-conditioned ResUNet (`4690176_0_1770580046_train_base_16`)

#### Artifact

- Inference artifact format: ONNX (`model_infer.onnx`)
- Current release reference: `v2026.02.19`
- Related training metadata (when packaged): `train_config.json`

#### Inference-Engine Boundary Contract
see below for draft assertion.

- Tensor names:
  - inputs: `depth_lr`, `dem_hr`
  - output: `depth_hr_pred`
- Tensor layout and dtype:
  - NHWC `float32`, single channel
  - `depth_lr`: `[N, 32, 32, 1]`
  - `dem_hr`: `[N, 512, 512, 1]`
  - `depth_hr_pred`: `[N, 512, 512, 1]`
  - `N` is dynamic (MVP runs with `N=1`)
- Geometry:
  - fixed scale `16` (`512 / 32`)
  - output H/W must match `dem_hr` H/W
- Value-domain at boundary entry:
  - finite values only
  - inputs normalized to `[0, 1]`
- Value-domain at boundary exit:
  - output remains normalized/log-space depth in `[0, 1]` before inverse transform

#### Model-Specific Pre/Post Rules
Reference: `others/inference_inline.ipynb`.

- Read model defaults from `train_config.json` when present:
  - `upscale`, `input_shape`, `max_depth`, DEM stats
- infill nodata:
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
- Normalize depth and DEM to `[0, 1]` before engine execution.
- Invert normalized output to depth units after mosaicking and final post-processing.
- re-apply  nodata mask







### Custom Assertion Function (draft)

```python
from typing import Mapping
import numpy as np


def assert_0014_inference_engine_contract(
    *,
    feed_dict: Mapping[str, np.ndarray],
    output_arr: np.ndarray | None = None,
    scale: int = 16,
    lr_tile: int = 32,
    hr_tile: int = 512,
) -> None:
    """Validate model-boundary tensors for ADR-0014.

    Parameters
    ----------
    feed_dict:
        Mapping of ONNX input name to batched NHWC arrays.
        Must include `depth_lr` and `dem_hr`.
    output_arr:
        Optional model output tensor (`depth_hr_pred`) to validate.
        If passed, must be NHWC float32 and shape-compatible with HR tile geometry.
    scale:
        Expected HR/LR upscale ratio.
    lr_tile:
        Expected LR tile side length.
    hr_tile:
        Expected HR tile side length.
    """
    assert scale > 0 and lr_tile > 0 and hr_tile > 0
    assert hr_tile == lr_tile * scale, f"hr_tile={hr_tile} must equal lr_tile*scale={lr_tile*scale}"

    for req_name in ("depth_lr", "dem_hr"):
        assert req_name in feed_dict, f"missing required input: {req_name}"

    depth_lr = np.asarray(feed_dict["depth_lr"])
    dem_hr = np.asarray(feed_dict["dem_hr"])

    assert depth_lr.dtype == np.float32, f"depth_lr dtype must be float32, got {depth_lr.dtype}"
    assert dem_hr.dtype == np.float32, f"dem_hr dtype must be float32, got {dem_hr.dtype}"
    assert depth_lr.ndim == 4, f"depth_lr must be NHWC rank-4, got shape={depth_lr.shape}"
    assert dem_hr.ndim == 4, f"dem_hr must be NHWC rank-4, got shape={dem_hr.shape}"

    assert depth_lr.shape[-1] == 1, f"depth_lr C dim must be 1, got shape={depth_lr.shape}"
    assert dem_hr.shape[-1] == 1, f"dem_hr C dim must be 1, got shape={dem_hr.shape}"

    assert depth_lr.shape[1:3] == (lr_tile, lr_tile), (
        f"depth_lr spatial shape must be {(lr_tile, lr_tile)}, got {depth_lr.shape[1:3]}"
    )
    assert dem_hr.shape[1:3] == (hr_tile, hr_tile), (
        f"dem_hr spatial shape must be {(hr_tile, hr_tile)}, got {dem_hr.shape[1:3]}"
    )
    assert depth_lr.shape[0] == dem_hr.shape[0], (
        f"batch mismatch depth_lr={depth_lr.shape[0]} dem_hr={dem_hr.shape[0]}"
    )

    assert np.isfinite(depth_lr).all(), "depth_lr must contain only finite values"
    assert np.isfinite(dem_hr).all(), "dem_hr must contain only finite values"
    assert float(depth_lr.min()) >= 0.0 and float(depth_lr.max()) <= 1.0, "depth_lr must be in [0,1]"
    assert float(dem_hr.min()) >= 0.0 and float(dem_hr.max()) <= 1.0, "dem_hr must be in [0,1]"

    if output_arr is not None:
        out = np.asarray(output_arr)
        assert out.dtype == np.float32, f"depth_hr_pred dtype must be float32, got {out.dtype}"
        assert out.ndim == 4, f"depth_hr_pred must be NHWC rank-4, got shape={out.shape}"
        assert out.shape[0] == depth_lr.shape[0], (
            f"output batch mismatch out={out.shape[0]} inputs={depth_lr.shape[0]}"
        )
        assert out.shape[1:3] == (hr_tile, hr_tile), (
            f"depth_hr_pred spatial shape must be {(hr_tile, hr_tile)}, got {out.shape[1:3]}"
        )
        assert out.shape[-1] == 1, f"depth_hr_pred C dim must be 1, got shape={out.shape}"
        assert np.isfinite(out).all(), "depth_hr_pred must contain only finite values"
```
 

