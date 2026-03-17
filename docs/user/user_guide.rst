User Guide
==========

Introduction
------------

``floodsr`` is a flood-depth **resolution enhancement** tool.
Or a super-resolution (SR) tool in machine-speak.
It takes a low-resolution depth raster as input and reconstructs higher-resolution output, using terrain (DEM) context.

Models
------

ResUNet_16x_DEM
~~~~~~~~~~~~~~~

This model reconstructs high-resolution flood depth from lower-resolution depth
input with DEM guidance.

Training
~~~~~~~~

Training pairs low-resolution depth with high-resolution terrain context.
Depth and DEM inputs are normalized for stable learning, and deterministic data
splits keep runs comparable. The optimizer uses Adam with gradient clipping and
scheduled learning-rate changes.

Inference
~~~~~~~~~

Inference mirrors training assumptions.
Inputs are validated and tiled, predictions are produced per tile, overlapping
outputs are feathered together, and the final surface is converted back to depth
units on the target grid.
