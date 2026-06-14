# ADR-0011: Placement of Project-Wide Parameters and Defaults

## Context

The `floodsr` package requires several categories of project-wide values, including:

- Immutable runtime constants (e.g., internal defaults, schema versions)
- User-configurable defaults (e.g., backend selection, verbosity, cache overrides)
- Library-specific I/O defaults (e.g., GDAL/GeoPackage driver options)

Historically, these were colocated in a single `parameters.py` module at the project root. This approach leads to unclear ownership, hidden coupling, and difficulty distinguishing between hard-coded constants and user-level configuration.

Because `floodsr` is a pip-installable CLI tool (intended for use in isolated environments and potentially via QGIS), configuration must be explicit, overrideable, and free of mutable global state.

## Decision

Project-wide values will be separated by stability and ownership into distinct modules:

### 1. Immutable Runtime Constants

**Definition**  
Values that are safe to change only via versioned release.

**Examples**
- Manifest schema version
- Internal default tile sizes
- Built-in nodata sentinel
- Internal directory names

**Location**
```
floodsr/constants.py
```
**Rules**
- Must be treated as immutable.
- No environment-dependent paths.
- No user preferences.
- No mutation at runtime.

---

### 2. Library-Specific Defaults

**Definition**  
Driver or engine options tied to specific I/O libraries (GDAL, rasterio, pyogrio, etc.).

**Examples**
- Default GeoTIFF profile
- Default GPKG dataset/layer options
- Compression settings

**Location**
```

floodsr/io/raster_defaults.py
floodsr/io/vector_defaults.py

````

**Rules**
- These are not global parameters.
- Must be close to the I/O code that consumes them.
- Treated as default profiles, not mutable shared dicts.
- Writers must merge overrides explicitly:

```python
profile = {**DEFAULT_PROFILE, **(user_profile or {})}
```

---

## Explicitly Rejected

### Root-Level `parameters.py`

Rejected because:

* Encourages mixing unrelated concerns.
* Promotes hidden global coupling.
* Blurs distinction between constants and configuration.
* Encourages mutation of shared dicts.

### `PROJECT_ROOT = Path(__file__).parent`

Rejected for runtime use in installed packages:

* Resolves to `site-packages` when installed.
* Not appropriate for writable state.
* All writable state must use `platformdirs`.

---

## Consequences

* Clear separation of stability domains.
* Improved testability (no hidden global state).
* Safe pip installation behavior.
* Predictable override semantics.
