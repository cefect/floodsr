"""Execution provider helpers for engine diagnostics."""

import importlib.metadata as md


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
