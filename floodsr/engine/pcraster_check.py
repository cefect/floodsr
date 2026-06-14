"""PCRaster runtime probes used by CostGrow and doctor diagnostics."""

import importlib.metadata as md


_PCRASTER_INSTALL_HINT = (
    "PCRaster is required for CostGrow_Terrain. "
    "Use the extended conda environment with `pcraster` installed."
)


def _check_pcraster():
    """Import and return `pcraster`, or raise a helpful runtime error."""
    try:
        import pcraster
    except ImportError as exc:
        raise ImportError(_PCRASTER_INSTALL_HINT) from exc
    return pcraster


def get_pcraster_info() -> dict[str, object]:
    """Return PCRaster installation diagnostics without hard import requirements."""
    try:
        version = md.version("pcraster")
    except md.PackageNotFoundError:
        version = None

    try:
        module = _check_pcraster()
    except ImportError as exc:
        return {
            "installed": False,
            "version": version,
            "spreadzone_available": False,
            "error": str(exc),
        }

    return {
        "installed": True,
        "version": version,
        "spreadzone_available": hasattr(module, "spreadzone"),
        "error": None,
    }
