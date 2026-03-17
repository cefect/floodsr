Installation
============

A-bit-too-simple moonshot install:

.. code-block:: bash

   pipx install floodsr

pipx command not found? Check :ref:`system_requirements`.

.. _system_requirements:

System Requirements
---------------------
``floodsr`` was designed as a CLI-first Python package, so we recommend installing with `pipx <https://pipx.pypa.io/stable/>`_ to ensure environment isolation.
`pipx requirements <https://pipx.pypa.io/stable/installation/>`_ are currently `Python 3.9+ <https://realpython.com/installing-python/>`_ and pip (which  usually comes shipped with Ptyhon).
If you have a modern Python setup, installing pipx is easy:


.. code-block:: bash

   python -m pip install --user pipx
   python -m pipx ensurepath

If you see a warning about needing to do something for *PATH changes to take effect*, follow the instructions.


Basic Install
-----------------
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


Extended Install
----------------
For handling rasters too large for memory, floodsr uses GDAL backends.
To enable these features, install `floodsr` into an environment with `GDAL <https://gdal.org/en/stable/>`_.
The popular `conda <https://docs.conda.io/en/latest/>`_ package manager is the easiest way to do this.
The best way to install conda is via the open-source `Miniforge <https://github.com/conda-forge/miniforge?tab=readme-ov-file#install>`_ project, NOT the proprietary Anaconda distribution.


Once you have conda installed, use it to create a dedicated environment with GDAL, activate it, then install ``floodsr`` with ``pip`` into that same environment:

.. code-block:: bash

   conda create -n floodsr-gdal -c conda-forge python=3.12 gdal -y
   conda activate floodsr-gdal
   python -m pip install floodsr

Now the ``doctor`` command should report GDAL Python bindings as installed and VRT support as enabled:

.. code-block:: bash

   floodsr doctor

Good job. you deserve some hi-res flood rasters now.

