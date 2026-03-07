# GitHub Workflows

This directory contains CI workflows for this repository.

## `pipx-smoke.yml`

Purpose:
- Keep a fast packaging/entrypoint smoke gate via `pipx`.

What it does:
1. Checks out the repo.
2. Sets up Python 3.11.
3. Installs `pipx`.
4. Installs the project using:
   - `python -m pipx install --force .`
5. Runs a simple CLI smoke sequence:
   - `floodsr --help`
   - `floodsr doctor`
   - `floodsr models list`

## `build-install-strategy.yml`

Purpose:
- Build one wheel and one sdist.
- Validate the progressive core vs. extended install strategy from those built artifacts.

What it does:
1. Builds `dist/*` with `python -m build`.
2. Runs `twine check` on the artifacts.
3. Smoke tests the core wheel in isolated envs without installing GDAL bindings.
4. Smoke tests the extended wheel after installing system GDAL and matching Python GDAL bindings.

## `full-tests.yml`

Purpose:
- Run the full CI test suite.
- Exclude local-only sphinx-marked tests and all network-marked tests.

What it does:
1. Checks out the repo.
2. Sets up Python 3.11.
3. Installs system GDAL plus project/test dependencies for the extended path.
4. Runs:
   - `pytest -m "not sphinx and not network"`

Triggers:
- `pull_request`
- `push` to `main`
- `workflow_dispatch` (manual run)

## Configuration

Common edits in `.github/workflows/pipx-smoke.yml`:
- Python version: change `actions/setup-python` -> `python-version`.
- Platform: change `runs-on` (currently `ubuntu-latest`).
- Smoke install target: change the pipx install string from `.` as needed.
- Smoke commands: edit the `Smoke test CLI` step.

Common edits in `.github/workflows/full-tests.yml`:
- Python version: change `actions/setup-python` -> `python-version`.
- Platform: change `runs-on` (currently `ubuntu-latest`).
- Test dependencies: edit the `Install system GDAL` and `Install test dependencies` steps.
- Test selection: edit the `Run pytest suite` step.

## Running

From GitHub UI:
1. Open **Actions**.
2. Select **CI - pipx smoke**.
3. Click **Run workflow**.

Using GitHub CLI:
```bash
gh workflow run pipx-smoke.yml
```

## Interpreting failures

Typical failure buckets:
- Packaging/install errors: project metadata, dependency resolution, wheel build.
- CLI import errors: missing runtime deps or import-time assumptions.
- Command contract regressions: changed or removed CLI subcommands/options.
- Test tier selection drift: a network-dependent test is missing `@pytest.mark.network` and leaks into CI.
