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



# Test Strategy and CI/CD Gates

 
## Context

`floodsr` is a pip-installable, CLI-first package. We want:
- Fast, deterministic feedback in local development (e.g., VS Code).
- Reliable cross-platform verification on clean machines via GitHub Actions.
- A clear separation between fast unit tests and slower, higher-fidelity end-to-end checks.
- Occasional tests that require network access (pinned artifacts), without making all PR runs flaky.

We also want to keep tests organized by *module* (mirroring the package layout), while classifying tests by *tier* (unit / e2e / network).

## Decision

1. **All tests are written in `pytest`**, organized by module path, and classified using **markers**:
   - `unit`: fast, deterministic, no network, minimal filesystem.
   - `e2e`: CLI-level tests that exercise the pipeline end-to-end.
   - `network`: tests that require network access (e.g., downloading pinned weights/test data).

2. **Test organization mirrors modules**, not tiers:
   - `tests/<module_path>/test_*.py`
   - Markers determine tier; directory structure does not.

3. **Local development default** (VS Code / developer workflow):
   - Run unit tests only by default.
   - E2E and network tests are opt-in.

4. **CI/CD policy**:
   - **Pull Requests:** run unit + e2e (no network).
   - **Releases:** run the full test suite: unit + e2e + network.

## Rationale

- Mirroring module layout keeps ownership and navigation clear as the codebase grows.
- Markers provide flexible tier selection without duplicating directory structure.
 
## Consequences

- E2E tests must be kept small enough to run on PRs (or they will slow feedback loops).
- Network tests must:
  - Use pinned URLs and expected hashes.
  - Fail with actionable messages when downloads change or are unavailable.
- Developers must remember to run E2E/network tiers locally when changing pipeline behavior.

## Implementation Notes

### Markers
Add to `pyproject.toml` (or `pytest.ini`) marker registration:
- `unit`: fast, deterministic tests (default local run)
- `e2e`: end-to-end CLI/system tests
- `network`: requires network access for pinned artifacts

### Example structure
 