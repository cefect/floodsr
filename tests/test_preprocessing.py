"""Tests for preprocessing utilities."""

import numpy as np
import pytest

from floodsr.preprocessing import write_prepared_rasters


def test_write_prepared_rasters_outputs_exist_and_are_float32(
    synthetic_inference_tiles: dict,
    tmp_path,
    logger,
) -> None:
    """Prepared outputs should exist on disk and keep float32 inference arrays."""
    rasterio = pytest.importorskip("rasterio")
    prepared = write_prepared_rasters(
        depth_lr_fp=synthetic_inference_tiles["depth_lr_fp"],
        dem_hr_fp=synthetic_inference_tiles["dem_fp"],
        scale=16,
        out_dir=tmp_path,
        logger=logger,
    )

    assert prepared["depth_lr_prepared_fp"].exists()
    assert prepared["dem_hr_prepared_fp"].exists()
    with rasterio.open(prepared["dem_hr_prepared_fp"]) as ds:
        dem_array = ds.read(1)
    assert dem_array.dtype == np.float32
    assert dem_array.size > 0
