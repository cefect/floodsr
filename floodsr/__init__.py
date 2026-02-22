"""floodsr package."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Read installed package metadata so version stays tied to pyproject.toml.
    __version__ = version("floodsr")
except PackageNotFoundError:
    __version__ = "0+unknown"
