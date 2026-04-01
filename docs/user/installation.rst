.. _installation:

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

- :ref:`basic_install_cli`: best for operations and those comfortable with a command-line interface (CLI). 
- :ref:`basic_install_local_notebook`:  interactive use and visualization. 
- :ref:`basic_install_google_colab`: simplest setup, but subject to cloud limits.
 
 


.. _basic_install_cli: 

Command line (CLI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A command-line interface (CLI) means running commands in a terminal instead of clicking through a graphical app.
``floodsr`` was designed as a CLI-first Python package, so we recommend installing with `pipx <https://pipx.pypa.io/stable/>`__ for simple environment isolation.
For this, you need a working `Python 3.12+ <https://realpython.com/installing-python/>`_ and pip (which usually comes already shipped with Python).
After this, installing `pipx <https://pipx.pypa.io/stable/>`__ is easy:

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
If you also want to run the ``CostGrow_Terrain`` model, install `PCRaster <https://pcraster.geo.uu.nl/>`_ into that same environment.


.. _extended_install_cli:

Command line (CLI) - Extended
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The popular `conda <https://docs.conda.io/en/latest/>`_ package manager is the easiest way to build an isolated environment with GDAL.
The best way to install conda is via the open-source `Miniforge <https://github.com/conda-forge/miniforge>`_ project, NOT the `proprietary Anaconda distribution <https://www.theregister.com/2024/08/08/anaconda_puts_the_squeeze_on/>`_.
Once you have conda installed, use it to create a dedicated environment with GDAL, activate it, then install ``floodsr`` with ``pip`` into that same environment:

.. code-block:: bash

   conda create -n floodsr-gdal -c conda-forge python=3.12 gdal -y
   conda activate floodsr-gdal
   python -m pip install floodsr

If you want the ``CostGrow_Terrain`` model, install PCRaster into that same environment:

.. code-block:: bash

   conda install -n floodsr-gdal -c conda-forge pcraster -y

Then confirm the GDAL-backed environment is active:

.. code-block:: bash

   floodsr doctor


You should see something like *gdal_config_installed=True*.
If PCRaster is installed, ``floodsr doctor`` should also report *pcraster_installed=True*.

.. _extended_install_notebook:

Local notebook (Jupyter) - Extended
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Follow the same :ref:`extended_install_cli` instructions to set up your GDAL-backed ``floodsr`` environment, then install Jupyter into that same environment and register a dedicated kernel:

.. code-block:: bash

   conda activate floodsr-gdal # make sure you're in the GDAL-backed environment created above
   python -m pip install jupyterlab ipykernel matplotlib 
   python -m ipykernel install --user --name floodsr-gdal --display-name "Python (floodsr-gdal)"

Then launch Jupyter from that environment and select the "Python (floodsr-gdal)" kernel:

.. code-block:: bash

   python -m jupyter lab

Finally, confirm the GDAL-backed environment is active from the notebook:

.. code-block:: bash

   !floodsr doctor

.. _extended_install_hosted_notebook:

Hosted notebook (Colab) - Extended
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While Colab is nice for standard environments, GDAL is not standard... so the extended ``floodsr`` with GDAL install is finnicky and probably won't work. 
For the foolhardy, the below seems to be working as of March 2026, but this will probably break when Colab updates their base images so maybe don't waste your time:

.. code-block:: bash

   # setup an experimental GDAL environment in Colab
   !apt-get update -qq
   !apt-get install -y -qq gdal-bin libgdal-dev
   !pip install -q --upgrade pip
   !pip install -q "gdal[numpy]==$(gdal-config --version).*" rasterio geopandas pyproj shapely fiona

   # install floodsr
   !pip install -q floodsr


Uninstall
---------
Use the uninstall command that matches both your install mode and execution context.

Basic uninstall
^^^^^^^^^^^^^^^

- **Command line (CLI)** with ``pipx``:

.. code-block:: bash

   pipx uninstall floodsr

- **Local notebook (Jupyter)** with ``pip`` in the kernel environment:

.. code-block:: bash

   python -m pip uninstall floodsr
   jupyter kernelspec uninstall floodsr

- **Hosted notebook (Colab)** with ``pip`` in the runtime:

.. code-block:: bash

   !python -m pip uninstall -y floodsr

Extended uninstall
^^^^^^^^^^^^^^^^^^

- **Command line (CLI)** or **local notebook (Jupyter)** in the ``floodsr-gdal`` conda environment:

.. code-block:: bash

   conda deactivate
   conda env remove -n floodsr-gdal

- **Hosted notebook (Colab)** experimental GDAL setup:

.. code-block:: bash

   # easiest cleanup is to restart the runtime
   # Runtime > Restart session
 
