"""Execution provider helpers for engine diagnostics."""

import importlib.metadata as md
import shutil, subprocess


def get_onnxruntime_info() -> dict[str, object]:
    """Return ORT installation and provider diagnostics."""
    import onnxruntime as ort

    return {
        "installed": True,
        "version": md.version("onnxruntime"),
        "available_providers": list(ort.get_available_providers()),
    }


def get_rasterio_info() -> dict[str, object]:
    """Return rasterio installation diagnostics."""
    try:
        version = md.version("rasterio")
    except md.PackageNotFoundError:
        return {
            "installed": False,
            "version": None,
        }
    return {
        "installed": True,
        "version": version,
    }


def get_gdal_info() -> dict[str, object]:
    """Return GDAL installation diagnostics for CLI and workflow smoke checks."""
    gdal_config_path = shutil.which("gdal-config")
    gdal_config_version = None
    if gdal_config_path is not None:
        result = subprocess.run(
            [gdal_config_path, "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            gdal_config_version = result.stdout.strip() or None

    # Probe Python GDAL separately so core installs stay importable.
    try:
        from osgeo import gdal
    except ImportError:
        return {
            "python_bindings_installed": False,
            "python_bindings_version": None,
            "gdal_config_installed": gdal_config_path is not None,
            "gdal_config_version": gdal_config_version,
            "vrt_enabled": False,
        }

    version = None
    try:
        version = gdal.VersionInfo("--version")
    except Exception:
        pass

    return {
        "python_bindings_installed": True,
        "python_bindings_version": version,
        "gdal_config_installed": gdal_config_path is not None,
        "gdal_config_version": gdal_config_version,
        "vrt_enabled": True,
    }
