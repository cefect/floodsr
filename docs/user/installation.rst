Installation
============

The probably-too-simple moonshot install:

.. code-block:: bash

   pipx install floodsr

- what is going on? Start reading at :ref:`basic_install`
- pipx command not found? Check :ref:`basic_install_cli`
- Running in a Notebook? Jump to :ref:`basic_install_google_colab` or :ref:`basic_install_local_notebook`
- Running large rasters? See :ref:`extended_install`  

 
.. _basic_install:

Basic Install
-------------
``floodsr`` comes in two flavours: 

- **basic install**: for rasters less than ~2 billion pixels or ~25 sq. km. (assuming a 1m target resolution)
- **extended install**: for rasters too large for memory. If you think you need this (and are feeling brave), skip ahead to :ref:`extended_install`. 

As ``floodsr`` is a simple `pypi distributed python package <https://pypi.org/project/floodsr/>`_, it can be installed in a variety of contexts.
Here we provide instructions for the three most common:

- :ref:`basic_install_cli`: best for operations and those comfortable with CLI. 
- :ref:`basic_install_local_notebook`:  interactive use and visualization. 
- :ref:`basic_install_google_colab`: simplest setup, but subject to cloud limits.
 
 


.. _basic_install_cli: 

Command line (CLI)
^^^^^^^^^^^^^^^^^^

``floodsr`` was designed as a CLI-first Python package, so we recommend installing with `pipx <https://pipx.pypa.io/stable/>`__ for simple environment isolation.
For this, you need a working `Python 3.12+ <https://realpython.com/installing-python/>`_ and pip (which usually comes already shipped with Python).
After this, installing `pipx <https://pipx.pypa.io/stable/installation/>`__ is easy:

.. code-block:: bash

   python -m pip install --user pipx
   python -m pipx ensurepath

If you see a warning about needing to do something for *PATH changes to take effect*, follow the instructions from the warning.

Then install ``floodsr`` with pipx:

.. code-block:: bash

   pipx install floodsr

If you see a message like *installed package floodsr*, you're g2g and should have access to the ``floodsr`` command line (CLI), which you can use to validate the install.
Start with the help command to confirm the command line (CLI) is working:

.. code-block:: bash

   floodsr --help

You can also try the ``doctor`` command to echo the current environment and ``floodsr`` configuration:

.. code-block:: bash

   floodsr doctor

This should show the version and status of the backends used by ``floodsr``.


.. _basic_install_local_notebook:

Local notebook (Jupyter)
^^^^^^^^^^^^^^^^^^^^^^^^

Running ``floodsr`` in a local notebook environment is a nice way to process individual rasters interactively and visualize the results.
To do this, you need to have `Jupyter <https://jupyter.org/install>`_ installed in a `Python 3.12+ <https://realpython.com/installing-python/>`_ environment.

Check your Python and Jupyter setup by running the following in a terminal:

.. code-block:: bash

   python --version
   python -m jupyter --version

Then, you can install ``floodsr`` with ``pip`` and register a dedicated Jupyter kernel for it:

.. code-block:: bash
   
   python -m pip install floodsr ipykernel
   python -m ipykernel install --user --name floodsr --display-name "Python (floodsr)"

Then launch Jupyter from that environment:

.. code-block:: bash

   python -m jupyter lab

.. note::

   Alternatively, for a quick and lazy one-time use install (nice for tutorials), you can run the following from **inside** a running kernel (i.e., notebook cell):

   .. code-block:: bash

      %pip install floodsr

   But this can lead to messy environments and is not recommended for regular use.

Finally, confirm the CLI is available from the notebook:

.. code-block:: bash

   !floodsr doctor



.. _basic_install_google_colab:

Hosted notebook (Colab)
^^^^^^^^^^^^^^^^^^^^^^^

`Google Colab <https://colab.research.google.com/>`__  is a popular hosted notebook environment that provides a nice web-based interface with common libraries pre-installed, which makes it great for quick experiments or interactive tutorials. 
However, patching in new packages (like ``floodsr``) can be a bit tricky, so we don't recommend this for regular use.


To setup a Colab notebook for ``floodsr``, launch `Google Colab <https://colab.research.google.com/>`__  in your browser and login with your Google account.
Then, create a new notebook and add the following to the first cell:

.. code-block:: bash

   !python -m pip install -q floodsr

Finally, confirm the CLI is available from the notebook:

.. code-block:: bash

   !floodsr --help



.. _extended_install:

Extended Install
----------------
For handling rasters too large for memory, floodsr uses GDAL backends.
To enable these features, install `floodsr` into an environment with `GDAL <https://gdal.org/en/stable/>`_.
The popular `conda <https://docs.conda.io/en/latest/>`_ package manager is the easiest way to do this.
The best way to install conda is via the open-source `Miniforge <https://github.com/conda-forge/miniforge?tab=readme-ov-file#install>`_ project, NOT the `proprietary Anaconda distribution <https://www.theregister.com/2024/08/08/anaconda_puts_the_squeeze_on/>`_.


.. _extended_install_cli:

Command line (CLI)
^^^^^^^^^^^^^^^^^^

Once you have conda installed, use it to create a dedicated environment with GDAL, activate it, then install ``floodsr`` with ``pip`` into that same environment:

.. code-block:: bash

   conda create -n floodsr-gdal -c conda-forge python=3.12 gdal -y
   conda activate floodsr-gdal
   python -m pip install floodsr

Then confirm the GDAL-backed environment is active:

.. code-block:: bash

   floodsr doctor


.. _extended_install_notebook:

Local notebook (Jupyter)
^^^^^^^^^^^^^^^^^^^^^^^^
If you want to work through the tutorial notebooks, first complete :ref:`extended_install_cli`.

Then install and launch `Project Jupyter <https://jupyter.org/install>`_ from that same conda environment. Creating or activating a new environment from inside a running notebook kernel is not reliable, so do the setup first, then launch Jupyter or switch the notebook kernel to that environment.

After completing :ref:`extended_install_cli`, the typical Jupyter-specific steps are:

.. code-block:: bash

   conda install -n floodsr-gdal -c conda-forge matplotlib pip ipykernel jupyterlab -y
   conda run -n floodsr-gdal python -m ipykernel install --user --name floodsr-gdal --display-name "Python (floodsr-gdal)"
   conda run -n floodsr-gdal jupyter lab

This gives the notebook environment GDAL support, the plotting stack used in the tutorials, and a dedicated Jupyter kernel for ``floodsr`` work. Once Jupyter is running, open the tutorial and select the ``Python (floodsr-gdal)`` kernel before executing any cells.

For a worked example, see :doc:`notebooks/tutorial_2`.

The ``doctor`` command from :ref:`extended_install_cli` should report GDAL Python bindings as installed and VRT support as enabled.

.. _extended_install_hosted_notebook:

Hosted notebook (Colab)
^^^^^^^^^^^^^^^^^^^^^^^

The extended GDAL-backed workflow is not currently supported in hosted notebook environments such as Google Colab.

.. code-block:: bash

   # setup an experimental GDAL environment in Colab
   !apt-get update -qq
   !apt-get install -y -qq gdal-bin libgdal-dev
   !pip install -q --upgrade pip
   !pip install -q "gdal[numpy]==$(gdal-config --version).*" rasterio geopandas pyproj shapely fiona

   # install floodsr
   !pip install -q floodsr

 
