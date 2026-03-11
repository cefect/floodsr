# releasing/publishing

See `docs/dev/adr/0013-publishing.md` and `docs/dev/adr/0017-cicd-workflow-policy.md`.

# Development Environment Setup

## Running dev-setup.sh

The `dev-setup.sh` script sets up the development environment using Docker:

```bash
chmod +x dev-setup.sh && ./dev-setup.sh
```

### What it does

1. Checks prerequisites (docker, git, git-lfs, gh)
2. Authenticates with GitHub and exports `FLOODSR_GITHUB_TOKEN`
3. Fetches Git LFS test data
4. **Builds a Docker image for x86_64 (`linux/amd64`) regardless of host architecture**
5. Validates that key packages (numpy, rasterio, pydantic) can be imported inside the container

### Platform support

The Docker image is always built for x86_64 (`--platform linux/amd64`):
- **On x86_64 hosts**: Builds natively, then validates package imports
- **On ARM64 hosts** (Apple Silicon, M1/M2/M3 Macs): Uses Docker's QEMU emulation for cross-compilation, then validates package imports

This ensures consistency across platforms since pcraster only has x86_64 builds on conda-forge. The validation test runs on all platforms inside the x86_64 container to verify key packages are importable.

### After setup completes

Run an interactive dev shell:

```bash
docker run --rm -it \
  --entrypoint /bin/bash \
  -v "$PWD:/workspace" \
  -e FLOODSR_GITHUB_TOKEN="$(gh auth token)" \
  -w /workspace \
  cefect/floodsr:miniforge-dev-v0.9 -l
```

Then verify inside the container:

```bash
python -m floodsr.cli models list
python -m floodsr.cli models fetch ResUNet_16x_DEM
pytest -q tests/test_model_registry.py
```

# CI/CD Triggered
----------------------------

## setup (one time)

### local packaging tools

Use the devcontainer image for local release tooling:

```bash
code .devcontainer/main/devcontainer.json
python -m pip show setuptools setuptools-scm build twine
```

### GitHub repository

Configure the repository once on GitHub:

1. Ensure GitHub Actions is enabled for the repository.
2. Keep `.github/workflows/release.yml` at that exact filename because PyPI Trusted Publishing binds to the workflow filename.
3. Create the GitHub environment `testpypi`.
4. Create the GitHub environment `pypi`.
5. Optionally add required reviewers or wait timers to the `pypi` environment before stable releases.

No PyPI API tokens or `~/.pypirc` entries are required for the CI/CD release path.

### Trusted Publishers

Configure GitHub Actions Trusted Publishing in both TestPyPI and PyPI for the `floodsr` project:

1. Sign in to each index and open the project settings for `floodsr`.
2. Add a Trusted Publisher for GitHub Actions with:
   - GitHub owner: the repository owner/org
   - Repository name: `floodsr`
   - Workflow filename: `release.yml`
   - Environment name: `testpypi` on TestPyPI, `pypi` on PyPI
3. If the project does not yet exist on an index, create a pending publisher first and let the first trusted publish create the project.




## creating a release

`setuptools-scm` is the version source. Do not edit a static package version in `pyproject.toml`.

### pre-release to TestPyPI

```bash
# 1) start from an up-to-date master branch
git checkout master
git pull --ff-only origin master

# 2) optional local sanity check before tagging
python -m build
python -m twine check dist/*

# 3) create and push an annotated pre-release tag
git tag -a v0.1.3rc1 -m "Release v0.1.3rc1"
git push origin v0.1.3rc1
```

This triggers `.github/workflows/release.yml`, which:
- verifies the tagged commit is reachable from `master`
- builds artifacts once
- runs unit and install-smoke validation
- publishes to TestPyPI
- creates or updates the GitHub Release from the same tag

### stable release to PyPI

```bash
# 1) start from an up-to-date master branch
git checkout master
git pull --ff-only origin master

# 2) create and push an annotated stable tag
git tag -a v0.0.2 -m "Release v0.0.2"
git push origin v0.0.2
```

This triggers the same release workflow, but stable tags publish to PyPI instead of TestPyPI.

## validating the trigger

After pushing a tag:

1. Open GitHub Actions and confirm the `Release` workflow started from the tag.
2. Confirm the `verify tag commit is on master` job passed.
3. Confirm the built version matches the tag in the build job logs.
4. Confirm the publish job targeted the correct index:
   - `testpypi` for `vX.Y.ZrcN`, `vX.Y.ZaN`, `vX.Y.ZbN`
   - `pypi` for `vX.Y.Z`
5. Confirm the GitHub Release exists for that same tag.

## quick post-publish checks

### TestPyPI

```bash
docker run --rm condaforge/miniforge3:25.3.1-0 bash -lc "
  set -euo pipefail &&
  export PIPX_HOME=/opt/pipx &&
  export PIPX_BIN_DIR=/usr/local/bin &&
  python -m pip install --upgrade pip pipx &&
  pipx install --index-url https://test.pypi.org/simple/ --pip-args='--extra-index-url https://pypi.org/simple' floodsr &&
  pipx runpip floodsr show floodsr &&
  floodsr doctor &&
  floodsr models list
"
```

### PyPI

```bash
python -m pip index versions floodsr
pipx install floodsr
floodsr doctor
floodsr models list
pipx uninstall floodsr
```
