User Guide
==========

Introduction
------------

``floodSR`` enhances depth raster resolution while preserving larger flood
structure and adding terrain-guided local detail.

Terminology
-----------

- LR depth: low-resolution depth input.
- HR output: high-resolution depth prediction.
- DEM: terrain input used to guide local structure.

How It Is Used
--------------

A typical workflow prepares LR depth and HR DEM rasters, runs model inference,
and compares outputs to a baseline and known hydraulic behavior.

See Also
--------

- :doc:`cli_reference`
- :doc:`models`
