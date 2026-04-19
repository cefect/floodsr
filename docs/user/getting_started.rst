Getting Started
===============

This page contains minimal info to get you up and flooding with ``floodsr``.
For more detailed information, see the :doc:`user_guide`.
For common questions, see :doc:`faq`.
For an interactive walkthrough, see :doc:`Tutorial 1 <notebooks/tutorial_1>`.

What Is ``floodsr``?
--------------------

``floodsr`` is a flood-depth **resolution enhancement** tool.
Or a super-resolution (SR) tool in machine-speak.
It takes a low-resolution depth raster as input and reconstructs higher-resolution output, using terrain (DEM) context.




Downloading test data
---------------------

Before running commands, it's nice to have some data to play with.
If you don't have your own data yet, you can download a test tile from the project.

To download manually, browse to `this release <https://github.com/cefect/floodsr/releases/tag/v0.0.3>`_ and download the assets into your current working directory.

Alternatively, ``bash`` users with ``curl`` can run:

.. code-block:: bash

   curl -L -O https://github.com/cefect/floodsr/releases/download/v0.0.3/hires002_dem.tif -O https://github.com/cefect/floodsr/releases/download/v0.0.3/lowres032.tif


Use
---

Here we give a quick intro on setting up a model and using it to enhance a flood raster.

Install
^^^^^^^^^^^

``floodsr`` was designed as a command-line interface (CLI)-first Python package, so we recommend installing with `pipx <https://pipx.pypa.io/stable/>`_ to ensure environment isolation:

.. code-block:: bash

   pipx install floodsr

For more detailed installation instructions, see :doc:`installation`.
After install, a quick sanity check is:

.. code-block:: bash

   floodsr doctor



Model Setup
^^^^^^^^^^^

``floodsr`` currently exposes both a downloaded machine-learning model
(``ResUNet_16x_DEM``) and a built-in rules-based model (``CostGrow_Terrain``).
List the available model versions with:

.. code-block:: bash

   floodsr models list

For the machine-learning backend, fetch the weights once and reuse them from cache:

.. code-block:: bash

   floodsr models fetch ResUNet_16x_DEM

``CostGrow_Terrain`` does not require downloaded weights, but it does require the
extended install with PCRaster. See :doc:`installation` if you want to use that model.

Now you're ready to enhance some flood rasters.

Enhance to High Resolution
^^^^^^^^^^^^^^^^^^^^^^^^^^

The primary tool in ``floodsr`` is the ``tohr`` command or *to high resolution*.
This ingests a low-resolution flood hazard raster and a high-resolution DEM.
This high-resolution DEM can either be specified as a local file or fetched from the `HRDEM Mosaic <https://open.canada.ca/data/en/dataset/0fe65119-e96e-4a57-8bfe-9d9245fba06b>`_ data source (for locations in Canada of course).

To enhance to high resolution, fetching the DEM from the HRDEM Mosaic, try:

.. code-block:: bash

   floodsr tohr --in lowres032.tif --fetch-hrdem

Alternatively, specify your own local DEM file:

.. code-block:: bash

   floodsr tohr --in lowres032.tif --dem hires002_dem.tif

To run the built-in CostGrow model explicitly, use the same command with a model version:

.. code-block:: bash

   floodsr tohr --in lowres032.tif --dem hires002_dem.tif --model-version CostGrow_Terrain

For more details, see the :doc:`user_guide`.
