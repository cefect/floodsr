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

``floodsr`` was designed as a CLI-first Python package, so we recommend installing with `pipx <https://pipx.pypa.io/stable/>`_ to ensure environment isolation:

.. code-block:: bash

   pipx install floodsr

For more detailed installation instructions, see :doc:`installation`.
After install, a quick sanity check is:

.. code-block:: bash

   floodsr doctor



Model Setup
^^^^^^^^^^^

Before using the machine-learning backend, we need to fetch the model weights.
This only needs to be done once, and the weights will be cached for future runs.

List available model versions:

.. code-block:: bash

   floodsr models list

You should see `ResUNet_16x_DEM` in the list, which is the only model currently available.

Fetch a model by version into the default cache:

.. code-block:: bash

   floodsr models fetch ResUNet_16x_DEM

Now you're ready to enhance some flood rasters!

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

For more details, see the :doc:`user_guide`.
