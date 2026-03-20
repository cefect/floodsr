# ADR-0002: Packaging and Installation Strategy

End users need a clean install path that does not pollute host environments, while still making room for features that truly depend on GDAL.

## background

There are two packaging constraints that drive the install strategy:

1. **Multiple ORT versions on the system can cause undefined behavior**

* ORT’s build docs explicitly warn that if multiple ORT versions are installed and library search paths are involved, ORT can find the wrong libraries and behave unpredictably.

2. **Some `floodsr` features have a real GDAL dependency**

* Commands that build or manipulate VRTs, and any future direct-GDAL features, require system GDAL plus matching Python bindings. (rasterio doesnt have vrt functions)


Because of that, the package should expose one progressive capability model rather than presenting all commands as equally available in every install.

## decision

- Ship one Python package providing:
  - Python library: `floodsr`
  - CLI entrypoint: `floodsr`
- Adopt a **progressive capability model** with two capability tiers:
  - **Basic**
    - Must not require `osgeo.gdal`.
    - Must not require system GDAL.
    - Assumes the user already has a working system Python for the target execution context.
    - For command line (CLI) use, the recommended install path is `pipx`.
    - For local notebook (Jupyter) use, the recommended install path is `pip` into the environment backing the notebook kernel.
    - For hosted notebook (Colab) use, installs happen inside the managed runtime with `pip`.
    - Is supported via `pip` and `pipx` on wheel-supported platforms.
    - Provides the default CLI and library surface for users who only need the non-VRT path.
    - VRT-dependent commands are unavailable.
    - Lower bounds in `pyproject.toml` should match one explicitly tested minimum core stack rather than a broad historical compatibility promise.
  - **Extended**
    - Is a conda-managed environment recipe rather than a PyPI extra.
    - Requires GDAL to be installed in the target conda environment before `pip install floodsr`.
    - Requires matching Python bindings in that same environment.
    - Enables VRT-dependent commands and any future direct-GDAL features.
- Do not publish a `pyproject.toml` GDAL extra because the Python GDAL binding version must track the environment-provided GDAL version.
- Keep the tested minimum core stack in a constraints file and validate it in CI.
- Treat the extended path as a pinned conda stack, not an open-ended dependency range.
- Keep publishing target as PyPI/TestPyPI wheels and source distributions.
- Validate published artifacts against the capability level they claim to support.
- Treat each documented install path as a support contract that should have a matching install-proof case.
- Prove documented Unix install paths from built release artifacts in isolated containerized environments that best match user expectations for that path.
- Keep current platform and execution-context support boundaries in `ADR-0007`.


## rationale

This project needs both a simple default install and a higher-capability path for GDAL-backed workflows.

Without this split we would need to either:
- break the simple-install goal for the default user path
- drop GDAL-backed workflows that depend on environment-provided GDAL and matching Python bindings
- or treat all installs as equally supported even though they have materially different dependency constraints

The basic-versus-extended split keeps the package surface unified while making the capability boundary explicit.

## consequences

- The package name and primary entrypoints stay simple: users always install `floodsr`.
- Basic remains the lowest-friction path for the default CLI and library experience.
- Basic documentation must state the preconditions separately from the package install itself: working Python for all local contexts, plus a working Jupyter setup for local notebook use.
- Extended remains available for GDAL-backed workflows without turning GDAL into an unreliable PyPI extra.
- Documentation and testing must clearly distinguish capability tier from platform and execution-context support.

## related decisions

- See `ADR-0007` for platform and execution-context support policy.
- See `ADR-0017` for CI/CD workflow policy.
