Model: 16x DEM-Conditioned ResUNet
==================================

This model is designed to recover high-resolution flood depth from lower-resolution
depth input, while using terrain information (DEM) as a guide. In practical terms,
it starts with a coarse depth map and uses land-surface shape to reconstruct a much
finer result.

Training
--------

During training, the model sees paired examples of low-resolution depth and
high-resolution terrain context. The training setup is built to be stable and
repeatable:

- Input depth values are clipped, transformed, and scaled to keep learning stable.
- DEM values are normalized tile-by-tile so terrain guidance is consistent across scenes.
- Data splits are deterministic, so repeated runs stay comparable.
- Optional augmentation (flips and rotations) is applied only to training samples.
- Optimization uses Adam with gradient clipping and a two-stage learning-rate schedule.
- Progress is tracked with quality metrics that reflect both overall accuracy and
  flood-specific behavior.

The overall goal in training is to learn both broad flood structure and fine local
detail, while staying robust across varied terrain.

Inference
---------

At inference time, the workflow mirrors the training assumptions so predictions remain
consistent:

1. Inputs are validated and normalized using the same model settings used at training
   time.
2. Large rasters are split into aligned tiles so the model can process them reliably.
3. Tiles are predicted in batches, with caching to avoid repeated work.
4. Overlapping tile outputs are feathered and stitched into a seamless full-scene map.
5. The final prediction is converted back to depth units, clipped to valid ranges, and
   resampled to the target output grid.

This produces a high-resolution depth surface that is visually coherent across tile
boundaries and faithful to both the input flood signal and local terrain structure.
