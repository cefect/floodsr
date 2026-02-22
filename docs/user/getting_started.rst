Getting Started
===============

Use this page for first contact with ``floodSR``.

What Is floodSR?
----------------

``floodSR`` is a flood-depth super-resolution tool.
It takes low-resolution depth input and reconstructs higher-resolution output,
using terrain (DEM) context to improve local detail.

Installation (Quick)
--------------------

Recommended first install:

.. code-block:: bash

   pipx install floodsr

See :doc:`installation` for full installation paths.

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
