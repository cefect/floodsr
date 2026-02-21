# ADR-0006: Testing and Test Data Contract

- Use `pytest`.
- Keep tests human-readable and behavior-oriented.
- Prefer fewer tests with clear intent over many overlapping tests.
- Keep test data in `tests/data` and keep artifacts small enough for quick unit runs.
- Use shared fixtures in `tests/conftest.py` for common setup.
- Store per-case metadata in `tests/data/<case>/case_spec.json`.
- Do not use `metrics.json` for case metadata.
- Put provenance/source details in `tests/data/<case>/readme.md` under `## Provenance`.
- Keep case compatibility/applicability switches under `flags` in `case_spec.json`.
- Include `flags.in_hrdem` when a case depends on HRDEM-specific behavior.
- wire in logging fixture to capture logs during test runs (where the function expects it).
- conclude test with a simple print statement to confirm test completion and provide a clear signal in test output.

`case_spec.json` should follow this contract:

```json
{
  "model": {
    "version": "4690176_0_1770580046_train_base_16",
    "file_name": "model_infer.onnx",
    "compatible": true
  },
  "inputs": {
    "lowres_fp": "lowres032.tif",
    "dem_fp": "hires002_dem.tif",
    "truth_fp": "hires002.tif"
  },
  "expected": {
    "precision": 3,
    "metrics": {
      "mase_m": 0.0589,
      "rmse_m": 0.1060,
      "ssim": 0.6654
    }
  },
  "flags": {
    "in_hrdem": false
  }
}
```

Test suite should follow this structure:

- `tests/test_inference_regression.py` should contain one parameterized regression test over all `case_spec.json` cases.
- Inference regression should assert output dtype, non-empty output, and expected metrics.
- Special-case behavior should branch only on explicit `flags`.
 