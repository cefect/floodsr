"""Tests for preprocessing utilities."""

from pathlib import Path

import numpy as np
import pytest

from floodsr.preprocessing import write_prepared_rasters


def test_write_prepared_rasters_creates_disk_outputs(
    synthetic_inference_tiles: dict,
    tmp_path: Path,
    logger,
) -> None:
    """Prepared outputs should be written to disk when preprocessing runs."""
    pytest.importorskip("rasterio")

    prepared = write_prepared_rasters(
        depth_lr_fp=synthetic_inference_tiles["depth_lr_fp"],
        dem_hr_fp=synthetic_inference_tiles["dem_fp"],
        scale=16,
        out_dir=tmp_path,
        logger=logger,
    )

    assert prepared["depth_lr_prepared_fp"].exists()
    assert prepared["dem_hr_prepared_fp"].exists()


def test_write_prepared_rasters_write_float32_arrays(
    synthetic_inference_tiles: dict,
    tmp_path: Path,
    logger,
) -> None:
    """Prepared rasters should retain float32 arrays for inference."""
    rasterio = pytest.importorskip("rasterio")
    prepared = write_prepared_rasters(
        depth_lr_fp=synthetic_inference_tiles["depth_lr_fp"],
        dem_hr_fp=synthetic_inference_tiles["dem_fp"],
        scale=16,
        out_dir=tmp_path,
        logger=logger,
    )

    with rasterio.open(prepared["dem_hr_prepared_fp"]) as ds:
        dem_array = ds.read(1)
    assert dem_array.dtype == np.float32
    assert dem_array.size > 0
