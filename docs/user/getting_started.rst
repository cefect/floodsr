Getting Started
===============

This page contains minimial info to get you up and flooding with ``floodSR``.
For more detailed information, see the :doc:`user_guide`.

What Is floodSR?
----------------

``floodSR`` is a flood-depth :resolution enhancement: tool. 
Or a super-resolution (SR) tool in machine-speak.
It takes low-resolution depth input and reconstructs higher-resolution output,
using terrain (DEM) context to improve local detail.

Installation (Quick)
--------------------

Recommended first install:

.. code-block:: bash

   pipx install floodsr

This is the core install. See :doc:`installation` for the extended GDAL-backed path.

Quickstart
----------

Run a minimal ``tohr`` pass with the test tile:

.. code-block:: bash

   floodsr tohr \
     --in tests/data/2407_FHIMP_tile/depth_lr.tif \
     --dem tests/data/2407_FHIMP_tile/dem_hr.tif \
     --out tests/data/2407_FHIMP_tile/depth_sr.tif

FAQ
---

This section is intentionally short for now and will expand with user questions.
