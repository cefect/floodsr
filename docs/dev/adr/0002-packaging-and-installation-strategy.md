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
- Adopt a **progressive capability model** with two installation modes:
  - **Core install**
    - Uses pure-Python or wheel-friendly dependencies only.
    - Must not require `osgeo.gdal`.
    - Provides the default CLI and library surface for users who only need the non-GDAL path.
    - VRT-dependent commands are unavailable.
  - **Extended install**
    - Requires system GDAL and matching Python bindings.
    - Enables VRT-dependent commands and any future direct-GDAL features.
- Keep publishing target as PyPI/TestPyPI wheels and source distributions.
- Validate published artifacts against the capability level they claim to support.


yes, this is complicated... but we spent a lot of time building the in-memory HRDEM fetcher.
So we either:
- break our **install must be simple** rule
- revert and have no in-memory HRDEM fetcher (could do some more testing with `merge(mem_limit....)`.. but ran out of time, and this won't support pre-filter)
- or we do the work to support both install paths and validate them properly.


## development container strategy

The development and deployment Docker images are always built for **x86_64 (`linux/amd64`)**:

1. **pcraster availability**: pcraster, a key dependency, only has x86_64 builds on conda-forge
2. **Cross-platform consistency**: Using `docker buildx build --platform linux/amd64` ensures the same image regardless of host architecture
3. **Host platform support**:
   - x86_64 hosts: builds natively
   - ARM64 hosts (Apple Silicon): uses QEMU emulation transparently via Docker
4. **Lock files**: conda lock files are platform-specific (x86_64); created during image build and verified on x86_64 hosts only

## deployment strategy

See `ADR-0017` for CI/CD workflow policy.

## CI/CD summary

See `ADR-0017` for CI/CD workflow policy.
