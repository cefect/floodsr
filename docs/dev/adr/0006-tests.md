# ADR-0006: Testing and Test Data Contract

See `ADR-0017` for CI/CD workflow policy.

## Context

- `floodsr` is a pip-installable, CLI-first package.
- Tests should stay human-readable, behavior-oriented, and fast enough for local development while still supporting reliable cross-platform CI and a smaller set of slower end-to-end and network checks.
- Test layout should mirror the package structure, not test tiers.
- Test data should stay small enough for quick fast-test runs.
- Shared fixtures should live in `tests/conftest.py`.
- Per-case metadata should live in `tests/data/<case>/case_spec.json`, with provenance and source details in `tests/data/<case>/readme.md` under `## Provenance`.

## Decision

- Use `pytest` throughout and prefer fewer clear tests over overlapping coverage.
- Organize tests by module path as `tests/<module_path>/test_*.py`; classify tiers with markers, not directories.


### regression tests
- `tests/test_tohr_regression.py` should contain one parameterized regression test over all `case_spec.json` cases.
- Parameterize ToHR regression by the human-readable run labels under `expected`; each run label defines CLI-style `params` and expected `metrics`.
- Each ToHR regression case should assert output dtype, non-empty output, and expected metrics, then print a simple completion message.


### Marks

- `fast`: fast, deterministic tests.
- `e2e`: end-to-end CLI or system tests. small enough for PR feedback loops.
- `network`: tests that require network access. must use pinned URLs and expected hashes, fail with actionable messages
- `notebook`: notebook execution tests for `docs/user/notebooks`. run these from the `dev` conda environment rather than the default `deploy` environment.
- `sphinx`: tests that require the documentation environment.
- `local`: local-only tests that depend on local fixture data.
- Do not use a `dev` mark; classify those tests as `local`.
- Register these marks in `pytest.ini`.
- Keep test module imports collection-safe across supported environments. If a module needs optional runtime dependencies that are absent from the docs environment, guard them with `pytest.importorskip(...)` or move the imports inside the tests/fixtures that need them so discovery can skip cleanly instead of erroring during collection.

### data-driven tests
- Keep compatibility or applicability switches under `flags` in `case_spec.json`, including `flags.in_hrdem` when a case depends on HRDEM-specific behavior.
- `case_spec.json` follows this contract:

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
