Installation
============

Progressive Capability Model
----------------------------

``floodsr`` ships with two install modes:

- ``floodsr``: the core install. This is the default package and does not require ``osgeo.gdal``.
- ``floodsr[extended]``: the GDAL-backed install for VRT-dependent workflows.

The HRDEM fetcher stays available in the core install, but it falls back to the
non-windowed GeoTIFF path when GDAL Python bindings are missing.

System Requirements
-------------------

Baseline requirements:

- Python 3.12+
- Linux or Windows for the documented install paths below
- Enough RAM/disk for raster tiling workflows
- For ``floodsr[extended]``: system GDAL plus matching Python GDAL bindings

The core install is exercised in CI on Ubuntu and Windows. The extended GDAL
install documented below is the Linux path currently documented and tested.

Install Core With pipx (Recommended)
------------------------------------

.. code-block:: bash

   pipx install floodsr

Reference: https://pipx.pypa.io/

Install Extended With pip (GDAL/VRT Features)
---------------------------------------------

Install system GDAL first, then install matching Python bindings before the
``extended`` extra. On Debian or Ubuntu that typically looks like:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y gdal-bin libgdal-dev
   python -m pip install "gdal==$(gdal-config --version)"
   python -m pip install "floodsr[extended]"

The Python GDAL version must match the system GDAL version reported by
``gdal-config --version``.

Install Core With pip (Advanced)
--------------------------------

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install floodsr
