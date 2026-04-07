# Tests

## Cross-References

- [`docs/dev/adr/0006-tests.md`](../docs/dev/adr/0006-tests.md): primary testing ADR for test strategy, markers, fixtures, and test data contracts.
- [`docs/dev/adr/0017-cicd-workflow-policy.md`](../docs/dev/adr/0017-cicd-workflow-policy.md): CI policy for test selection; CI runs `fast` and excludes `local`.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md): developer setup notes, `pytest` examples, and fixture-data prerequisites such as Git LFS assets.
- [`docs/dev/adr/0001-architecture-and-cli.md`](../docs/dev/adr/0001-architecture-and-cli.md): repo structure reference that reserves `tests/` as the package-aligned test location.



## patch dev environment
```bash
# patch floodsr cli
floodsr() { python -m floodsr.cli "$@"; }
export -f floodsr

# check the version of floodsr
floodsr --version
```

## Running The Suite In The Dev Environment

### core tests
```bash
# activate the default development environment
conda activate deploy
# move to the repository root
cd /workspace

# all tests (this is the VS Code UI default)
pytest -q -m "not sphinx" 2>&1 | tee "tests/log/pytest_$(date -u +%Y%m%d_%H%M%S)_all.log"


# fast local suite 
pytest -q -m "fast and not sphinx" 2>&1 | tee "tests/log/pytest_$(date -u +%Y%m%d_%H%M%S)_fast.log"

# CI test run
pytest -q -m "fast and not local and not sphinx" 2>&1 | tee "tests/log/pytest_$(date -u +%Y%m%d_%H%M%S)_ci.log"

# specific module
pytest -q tests/test_cli_models.py

 
 
```

### documentation tests
NOTE: these use a different .devcontainer
```bash
conda activate ???

# run sphinx/docs tests
pytest -q -m "sphinx"

```

### notebook tests
NOTE: these run in the `dev` conda environment, not `deploy`.

```bash
conda run -n dev pytest -q -m "notebook"
```

## Notes
- Marker definitions live in [`pytest.ini`](../pytest.ini).
 


## marker summary

| Test | `fast` | `e2e` | `network` | `sphinx` | `local` |
|---|---|---|---|---|---|
| `test_cache_paths.py::test_get_cache_dir_returns_created_path` | X |  |  |  |  |
| `test_checksums.py::test_compute_sha256_returns_hex_digest` | X |  |  |  |  |
| `test_checksums.py::test_verify_sha256_returns_expected_flag` | X |  |  |  |  |
| `test_cli_models.py::test_main_models_list_outputs_model_version` | X |  |  |  |  |
| `test_cli_models.py::test_resolve_log_level_from_cli_arguments` | X |  |  |  |  |
| `test_cli_models.py::test_main_models_fetch_prints_existing_path` | X |  |  |  |  |
| `test_cli_models.py::test_parse_models_fetch_progress_flags` | X |  |  |  |  |
| `test_cli_models.py::test_main_models_fetch_routes_errors_to_stderr` | X |  |  |  |  |
| `test_cli_models.py::test_main_version_reports_installed_package_version` | X |  |  |  |  |
| `test_cli_models.py::test_main_doctor_reports_runtime_diagnostics` | X |  |  |  |  |
| `test_cli_models.py::test_main_doctor_reports_runtime_diagnostics_json` | X |  |  |  |  |
| `test_cli_tohr.py::test_main_tohr_runs_data_driven_baseline_case` |  | X | X |  | X |
| `test_cli_tohr.py::test_main_tohr_runs_in_hrdem_flagged_case` |  | X | X |  | X |
| `test_cli_tohr.py::test_default_output_path_uses_cwd_and_input_stem` | X |  |  |  | X |
| `test_cli_tohr.py::test_resolve_tohr_model_spec_uses_cached_manifest_default` | X |  |  |  |  |
| `test_cli_tohr.py::test_parse_tohr_allows_fetch_hrdem_without_dem` | X |  |  |  |  |
| `test_cli_tohr.py::test_parse_tohr_allows_fetch_force_tiling_flag` | X |  |  |  |  |
| `test_cli_tohr.py::test_parse_tohr_allows_machine_json_only` | X |  |  |  |  |
| `test_cli_tohr.py::test_parse_tohr_cli_args_override_machine_json` | X |  |  |  |  |
| `test_cli_tohr.py::test_parse_tohr_rejects_dem_and_fetch_hrdem_together` | X |  |  |  |  |
| `test_cli_tohr.py::test_main_tohr_fetch_out_requires_fetch_hrdem` | X |  |  |  |  |
| `test_docs.py::test_docs_linkcheck_builds` |  |  |  | X | X |
| `test_engine_contracts.py::test_engine_provider_diagnostics_shape` | X |  |  |  |  |
| `test_engine_contracts.py::test_engine_base_is_abstract` | X |  |  |  |  |
| `test_engine_contracts.py::test_engine_base_contract_with_dummy_subclass` | X |  |  |  |  |
| `test_model_resunet.py::test_engine_ort_run_tile_contract` | X |  |  |  |  |
| `test_hrdem_mosaic.py::test_build_fetch_tile_grid_gdf_and_selection_mask_writes_geojson` | X |  |  |  | X |
| `test_hrdem_mosaic.py::test_download_hrdem_project_extent_for_data_case` | X |  | X |  | X |
| `test_hrdem_mosaic.py::test_fetch_hrdem_synthetic_cases` |  |  | X |  |  |
| `test_hrdem_mosaic.py::test_write_dem_from_asset_hrefs_synthetic_cases` | X |  |  |  |  |
| `test_hrdem_mosaic.py::test_write_dem_from_asset_hrefs_non_windowed_outputs_float32_non_empty` | X |  |  |  |  |
| `test_hrdem_mosaic.py::test_fetch_hrdem_data_case` |  |  | X |  |  |
| `test_lock_alignment.py::test_conda_lock_alignment` |  |  |  |  | X |
| `test_model_registry.py::test_list_models_returns_non_empty_records` | X |  |  |  |  |
| `test_model_registry.py::test_fetch_model_returns_cached_path` | X |  |  |  |  |
| `test_model_registry.py::test_fetch_model_fails_on_checksum_mismatch` | X |  |  |  |  |
| `test_model_registry.py::test_default_manifest_records_include_required_fields` | X |  |  |  |  |
| `test_model_registry.py::test_list_runnable_model_versions_match_worker_backed_manifest` | X |  |  |  |  |
| `test_model_registry.py::test_resolve_model_worker_class_returns_model_worker_type` | X |  |  |  |  |
| `test_model_registry.py::test_manifest_injected_bad_values_fail_fetch` | X |  |  |  |  |
| `test_model_registry.py::test_default_manifest_http_links_resolve` | X |  | X |  |  |
| `test_preprocessing.py::test_write_prepared_rasters_outputs_exist_and_are_float32` | X |  |  |  |  |
| `test_preprocessing.py::test_write_prepared_rasters_honors_crs_policy_for_mismatch` | X |  |  |  |  |
| `test_preprocessing.py::test_write_prepared_rasters_default_strict_rejects_crs_mismatch` | X |  |  |  |  |
| `test_preprocessing.py::test_write_platform_prepared_rasters_honors_crs_policy` | X |  |  |  |  |
| `test_preprocessing.py::test_write_dem_from_asset_hrefs_outputs_float32_non_empty` | X |  |  |  |  |
| `test_tohr_regression.py::test_tohr_regression_matches_case_spec_metrics` |  |  | X |  | X |
| `test_model_resunet.py::test_resunet_tohr_on_the_fly_synthetic_tiles` |  |  | X |  |  |
| `test_model_resunet.py::test_resunet_tohr_hard_windowed_tiles` |  |  | X |  |  |
| `test_model_costgrow.py::test_costgrow_tohr_uses_windowed_path_for_large_hard_rasters` | X |  |  |  |  |


# simple container tests

## miniforge3 [CLI]
```bash
# launch one-time use interactive shell
docker run --rm --init -it -v "$(mktemp -d):/tmp/work" -w /tmp/work -p 8888:8888 condaforge/miniforge3:25.3.1-0 bash

#isntall pipx
python -m pip install --user pipx
python -m pipx ensurepath
source ~/.bashrc

# install floodsr
pipx install floodsr



```

## miniforge3 [jupyter]
```bash
# launch one-time use interactive shell
docker run --rm --init -it -v "$(mktemp -d):/tmp/work" -w /tmp/work -p 8888:8888 condaforge/miniforge3:25.3.1-0 bash
 

# install jupyter
python -m pip install jupyterlab

# launch notebook
python -m jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --allow-root \
  --ServerApp.token='' \
  --ServerApp.password=''


# open in windows browser

# to run a tutorial notebook, probably open the docs website, right click "save as", then back on jupyter, "upload file"

```

## colab
```bash
us-docker.pkg.dev/colab-images/public/cpu-runtime:latest
```

## ubuntu:24.04
```bash
docker run --rm --init -it -v "$(mktemp -d):/tmp/work" -w /tmp/work ubuntu:24.04 bash
```
