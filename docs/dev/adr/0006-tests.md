# ADR-0006: Testing and Test Data Contract

- Use `pytest`.
- Keep tests human-readable and behavior-oriented.
- Prefer fewer tests with clear intent over many overlapping tests.
- Keep test data in `tests/data` and keep artifacts small enough for quick fast-test runs.
- Use shared fixtures in `tests/conftest.py` for common setup.
- Store per-case metadata in `tests/data/<case>/case_spec.json`.
- Put provenance/source details in `tests/data/<case>/readme.md` under `## Provenance`.
- Keep case compatibility/applicability switches under `flags` in `case_spec.json`.
- Include `flags.in_hrdem` when a case depends on HRDEM-specific behavior.
- conclude test with a simple print statement to confirm test completion and provide a clear signal in test output.

`case_spec.json` should follow this contract:

```json
{
  "inputs": {
    "lowres_fp": "lowres032.tif",
    "dem_fp": "hires002_dem.tif",
    "truth_fp": "hires002.tif"
  },
  "expected": {
    "ResUNet_16x_DEM_default": {
      "params": {
        "model_version": "ResUNet_16x_DEM",
        "window_method": "feather",
        "tile_overlap": 1
      },
      "metrics": {
        "precision": 3,
        "mase_m": 0.0589,
        "rmse_m": 0.1060,
        "ssim": 0.6654
      }
    },
    "ResUNet_16x_DEM_hard_tiles": {
      "params": {
        "model_version": "ResUNet_16x_DEM",
        "window_method": "hard",
        "tile_overlap": 0
      },
      "metrics": {
        "precision": 3,
        "mase_m": 0.0612,
        "rmse_m": 0.1097,
        "ssim": 0.6510
      }
    }
  },
  "flags": {
    "in_hrdem": false
  }
}
```

Test suite should follow this structure:

- `tests/test_tohr_regression.py` should contain one parameterized regression test over all `case_spec.json` cases.
- ToHR regression should parameterize by human-readable run labels under `expected`.
- Each run label should define CLI-style `params` and expected `metrics`.
- ToHR regression should assert output dtype, non-empty output, and expected metrics for each run label.




# Test Strategy

 
## Context

`floodsr` is a pip-installable, CLI-first package. We want:
- Fast, deterministic feedback in local development (e.g., VS Code).
- Reliable cross-platform verification on clean machines via GitHub Actions.
- A clear separation between fast tests and slower, higher-fidelity end-to-end checks.
- Occasional tests that require network access (pinned artifacts), without making local defaults or CI flaky.

We also want to keep tests organized by *module* (mirroring the package layout), while classifying tests by *tier* (`fast` / `e2e` / `network`).

## Decision

1. **All tests are written in `pytest`**, organized by module path, and classified using **markers**: see below. 

2. **Test organization mirrors modules**, not tiers:
   - `tests/<module_path>/test_*.py`
   - Markers determine tier; directory structure does not.

 

4. **CI/CD policy references**:
   - See `ADR-0017` for CI/CD workflow policy.

 

 
 
## Consequences

- E2E tests must be kept small enough to run on PRs (or they will slow feedback loops).
- Network tests must:
  - Use pinned URLs and expected hashes.
  - Fail with actionable messages when downloads change or are unavailable.
  - Be explicitly marked with `@pytest.mark.network` so they never leak into CI.
- Developers must remember to run E2E/network tiers locally when changing pipeline behavior.

## Implementation Notes

### Markers
Add to  `pytest.ini`  marker registration (NOTE: multiple markers can be used per test):
- `fast`: fast, deterministic tests (not exlcusive)
- `e2e`: end-to-end CLI/system tests
- `network`: requires network access  
- `sphinx`: documentation env (requires sphinx)
- `local`: local-only tests that depend on local fixture data
- do not register a `dev` marker; classify those tests as `local`

 
## Cross-References

- `ADR-0017` owns CI/CD workflow policy.
 
