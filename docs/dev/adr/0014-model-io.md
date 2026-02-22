# ADR-0001: inference model I/O
 

 

## Decision

- Artifact format: ONNX model files.
- Runtime for MVP: ONNX Runtime (CPU).
 
## inference-engine boundary
- Model I/O Contract (from `dev/infer_test_tiles.ipynb`) for `4690176_0_1770580046_train_base_16`
- NOTE: may have other contracts as additional models are added.

### contract
- Model I/O names:
  - inputs: `depth_lr`, `dem_hr`
  - output: `depth_hr_pred`
- Tensor layout and dtype:
  - all tensors are NHWC, `float32`, single channel (`C=1`)
  - `depth_lr`: `[N, 32, 32, 1]`
  - `dem_hr`: `[N, 512, 512, 1]`
  - `depth_hr_pred`: `[N, 512, 512, 1]`
  - `N` is dynamic; MVP notebook uses `N=1`
- Geometry constraints:
  - `depth_lr` and `dem_hr` are square tiles
  - output H/W must match `dem_hr` H/W
  - upscale ratio is `16` (512 / 32)
- Input preprocessing:
  - nodata replacement in notebook POC: nodata pixels are replaced with `0.0` before normalization
  - low-res depth (`depth_lr`) preprocessing:
    - clip to `[0, max_depth]`
    - apply `log1p(depth) / log1p(max_depth)`
    - clamp to `[0, 1]`
  - high-res DEM (`dem_hr`) preprocessing:
    - clip negatives to `0`
    - clip upper bound by DEM clip stat (`p_clip`)
    - min-max normalize with (`dem_min`, `dem_max`)
    - clamp to `[0, 1]`
    - prefer persisted training stats (`dem_stats`) from `train_config.json` when present
- Output postprocessing:
  - `depth_hr_pred` is normalized log-space
  - invert with `expm1(norm * log1p(max_depth))`
  - clamp back to `[0, max_depth]` in depth units
- Runtime/provider policy for MVP:
  - ORT session is created with `providers=["CPUExecutionProvider"]`
  - input names and static dimensions are validated against ORT session metadata before `session.run(...)`

## nodata normalization/handling
read nodata from DEM mask 
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

### pre-processing
see `docs/dev/adr/0009-preproccessing.md`



 