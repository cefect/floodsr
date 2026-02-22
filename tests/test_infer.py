"""End-to-end ToHR coverage for data-driven cases."""

from pathlib import Path

import numpy as np
import pytest

from conftest import TEST_TILE_CASES
from floodsr.dem_sources import fetch_dem
from floodsr.tohr import tohr


pytestmark = pytest.mark.e2e

_TOHR_E2E_CASES = [
    pytest.param(case_name, False, id=f"tohr_data_case_{case_name.lower()}_with_dem") for case_name in TEST_TILE_CASES
] + [
    pytest.param(
        "2407_FHIMP_tile",
        True,
        marks=pytest.mark.network,
        id="tohr_data_case_2407_fhimp_tile_fetch_hrdem",
    )
]


@pytest.mark.parametrize("tile_case, fetch_hrdem", _TOHR_E2E_CASES, indirect=["tile_case"])
def test_tohr_runs_data_driven_cases_end_to_end(
    inference_model_fp: Path,
    tile_case: dict,
    fetch_hrdem: bool,
    tmp_path: Path,
    logger,
) -> None:
    """Run ToHR end-to-end for each tracked tile case without metric matching."""
    pytest.importorskip("onnxruntime")
    rasterio = pytest.importorskip("rasterio")
    case_spec = tile_case["case_spec"]
    tile_dir = tile_case["tile_dir"]
    output_fp = tmp_path / f"{tile_case['case_name']}_tohr_e2e.tif"
    dem_hr_fp = tile_dir / case_spec["inputs"]["dem_fp"]

    # Cover one explicit HRDEM fetch path for a known in_hrdem case.
    if fetch_hrdem:
        pytest.importorskip("pystac_client")
        assert tile_case["case_name"] == "2407_FHIMP_tile"
        fetch_out_fp = tmp_path / f"{tile_case['case_name']}_fetched_hrdem.tif"
        try:
            fetch_result = fetch_dem(
                source_id="hrdem",
                depth_lr_fp=tile_dir / case_spec["inputs"]["lowres_fp"],
                output_fp=fetch_out_fp,
                logger=logger,
            )
        except Exception as exc:  # pragma: no cover - network/data service dependent
            pytest.skip(f"unable to fetch HRDEM for e2e test: {exc}")
        dem_hr_fp = fetch_result.dem_fp

    # Run one full ToHR pass against the data-driven case inputs.
    result = tohr(
        model_version=case_spec["model"]["version"],
        model_fp=inference_model_fp,
        depth_lr_fp=tile_dir / case_spec["inputs"]["lowres_fp"],
        dem_hr_fp=dem_hr_fp,
        output_fp=output_fp,
        logger=logger,
    )

    # Load output raster and validate high-level output contract only.
    with rasterio.open(result["output_fp"]) as ds:
        pred = ds.read(1)
    assert pred.dtype == np.float32
    assert pred.size > 0
