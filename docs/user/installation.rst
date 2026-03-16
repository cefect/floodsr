Installation
============

``floodsr`` supports two installation paths:

- Basic install: no ``osgeo.gdal``, no required system GDAL, supported via ``pip`` and ``pipx`` on wheel-supported platforms.
- Advanced install: Linux/conda-supported path for VRT workflows, with GDAL installed in the target conda environment before ``floodsr``.


Basic Install
-------------

The basic install is the recommended user path. It keeps the CLI isolated and does not require system GDAL or Python GDAL bindings.

Install ``pipx`` first if you do not already have it:

.. code-block:: bash

   python -m pip install --user pipx
   pipx ensurepath

Then install ``floodsr`` into its own isolated CLI environment:

.. code-block:: bash

   pipx install floodsr

Validate the install:

.. code-block:: bash

   floodsr --help
   floodsr doctor --json

The basic install supports the default CLI and non-VRT workflows. If GDAL is present elsewhere on the host, ``floodsr`` should still remain on the non-VRT path unless GDAL is installed inside the active Python environment.


Advanced Install
----------------

The advanced install enables GDAL-backed VRT workflows, including the tiled HRDEM fetch path that depends on ``osgeo.gdal``.

This path is supported for Linux users managing the environment with conda. Install conda first if needed:

- Conda install guide: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

Create a dedicated environment with GDAL from ``conda-forge``, activate it, then install ``floodsr`` with ``pip`` into that same environment:

.. code-block:: bash

   conda create -n floodsr-gdal -c conda-forge python=3.12 gdal -y
   conda activate floodsr-gdal
   python -m pip install --upgrade pip
   python -m pip install floodsr

Validate the advanced install:

.. code-block:: bash

   floodsr doctor --json

In the advanced install, ``floodsr doctor --json`` should report GDAL Python bindings as installed and VRT support as enabled.


Support Notes
-------------

- Basic installs are intended for ``pip`` and ``pipx`` on wheel-supported platforms.
- Advanced installs are documented and validated as a Linux/conda workflow.
- ``floodsr`` does not publish a GDAL extra in ``pyproject.toml`` because Python GDAL bindings must match the GDAL version already installed in the target environment.
