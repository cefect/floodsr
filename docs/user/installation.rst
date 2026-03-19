Installation
============

A bit-too-simple moonshot install:

.. code-block:: bash

   pipx install floodsr

pipx command not found? Check :ref:`system_requirements`.
Running in Google Colab? Jump to :ref:`basic_install_google_colab`.

.. _system_requirements:

System Requirements
-------------------
``floodsr`` was designed as a CLI-first Python package, so we recommend installing with `pipx <https://pipx.pypa.io/stable/>`__ to ensure environment isolation (except for Google Colab, where ``pip`` is the better fit. See :ref:`basic_install_google_colab`).
This project requires `Python 3.12+ <https://realpython.com/installing-python/>`_ and pip (which usually comes shipped with Python). 
Once you have a compatible Python install, installing `pipx <https://pipx.pypa.io/stable/installation/>`__ is easy.
If you have a modern Python setup, installing pipx is easy:


.. code-block:: bash

   python -m pip install --user pipx
   python -m pipx ensurepath

If you see a warning about needing to do something for *PATH changes to take effect*, follow the instructions.


Basic Install
-------------
Check pipx is installed and on the PATH:

.. code-block:: bash

   pipx --version

Then install ``floodsr`` with pipx:

.. code-block:: bash

   pipx install floodsr

If you see a message like *installed package floodsr*, you're g2g and should have access to the ``floodsr`` CLI, which you can use to validate the install.
Start with the help command to confirm the CLI is working:

.. code-block:: bash

   floodsr --help

You can also try the ``doctor`` command to echo the current environment and ``floodsr`` configuration:

.. code-block:: bash

   floodsr doctor

This should show the version and status of the backends used by ``floodsr``.


.. _basic_install_google_colab:

Basic Install (Google Colab)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Google Colab already gives you an isolated Python environment, so ``pipx`` is unnecessary there. Install the core package with ``pip`` in a notebook cell:

.. code-block:: bash

   python -m pip install floodsr

Then confirm the CLI is available:

.. code-block:: bash

   !floodsr --help


.. _extended_install:

Extended Install
----------------
For handling rasters too large for memory, floodsr uses GDAL backends.
To enable these features, install `floodsr` into an environment with `GDAL <https://gdal.org/en/stable/>`_.
The popular `conda <https://docs.conda.io/en/latest/>`_ package manager is the easiest way to do this.
The best way to install conda is via the open-source `Miniforge <https://github.com/conda-forge/miniforge?tab=readme-ov-file#install>`_ project, NOT the `proprietary Anaconda distribution <https://www.theregister.com/2024/08/08/anaconda_puts_the_squeeze_on/>`_.


Once you have conda installed, use it to create a dedicated environment with GDAL, activate it, then install ``floodsr`` with ``pip`` into that same environment:

.. code-block:: bash

   conda create -n floodsr-gdal -c conda-forge python=3.12 gdal -y
   conda activate floodsr-gdal
   python -m pip install floodsr


.. _extended_install_notebook:

Notebook
^^^^^^^^
If you want to work through the tutorial notebooks, install and launch `Project Jupyter <https://jupyter.org/install>`_ from the same conda environment as ``floodsr``. Creating or activating a new environment from inside a running notebook kernel is not reliable, so do the setup first, then launch Jupyter or switch the notebook kernel to that environment.

The typical pattern is:

.. code-block:: bash

   conda create -n floodsr-gdal -c conda-forge python=3.12 gdal matplotlib pip ipykernel jupyterlab -y
   conda run -n floodsr-gdal python -m pip install floodsr
   conda run -n floodsr-gdal python -m ipykernel install --user --name floodsr-gdal --display-name "Python (floodsr-gdal)"
   conda run -n floodsr-gdal jupyter lab

This gives the notebook environment GDAL support, the plotting stack used in the tutorials, and a dedicated Jupyter kernel for ``floodsr`` work. Once Jupyter is running, open the tutorial and select the ``Python (floodsr-gdal)`` kernel before executing any cells.

For a worked example, see :doc:`notebooks/tutorial_2`.

Now the ``doctor`` command should report GDAL Python bindings as installed and VRT support as enabled:

.. code-block:: bash

   floodsr doctor

Hosted Notebook (Google Colab)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # setup an experimental GDAL environment in Colab
   !apt-get update -qq
   !apt-get install -y -qq gdal-bin libgdal-dev
   !pip install -q --upgrade pip
   !pip install -q "gdal[numpy]==$(gdal-config --version).*" rasterio geopandas pyproj shapely fiona

   # install floodsr
   !pip install -q floodsr


Good job. You deserve some hi-res flood rasters now.
