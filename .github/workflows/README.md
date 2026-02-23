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

## `full-tests.yml`

Purpose:
- Run the full CI test suite.
- Exclude local-only sphinx-marked tests.

What it does:
1. Checks out the repo.
2. Sets up Python 3.11.
3. Installs project/test dependencies.
4. Runs:
   - `pytest -m "not sphinx"`

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
- Test dependencies: edit the `Install test dependencies` step.
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
