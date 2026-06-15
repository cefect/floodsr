# releasing/publishing

See `docs/dev/adr/0013-publishing.md` and `docs/dev/adr/0017-cicd-workflow-policy.md`.



# CI/CD Triggered
----------------------------

## setup (one time)
see below








## creating a release

`setuptools-scm` is the version source. Do not edit a static package version in `pyproject.toml`.

### branch and tag wiring
`.github/workflows/release.yml` only runs on pushed tags matching `v*`, and it first verifies that the tagged commit is reachable from `origin/master`.

This means:
- do your release prep on any branch you want
- merge the release commit to `master`
- push `master` to `origin`
- create the release tag on that pushed `master` commit
- push the tag

If the tag points to a commit that is not already on `origin/master`, the release workflow will fail before publishing.

### pre-release to TestPyPI

Accepted pre-release tag forms:
- `v0.1.3rc1`
- `v0.1.3a1`
- `v0.1.3b1`

These route to the `publish-testpypi` job in `.github/workflows/release.yml`.
NOTE: the testPyPi landing page shows the stable release by default (click ["release history"](https://test.pypi.org/project/floodsr/#history) to see pre-releases).

```bash
# 1) start from an up-to-date master branch
git checkout master
git pull --ff-only origin master

# 2) check existing tags and pick the next pre-release tag
git tag --sort=-v:refname | grep '^v' | head -n4

# 3) create the annotated pre-release tag on the current master commit
tag="v0.2.0a3"
git tag -a "$tag" -m "Release $tag"

# 4) push the branch first, then the tag
git push origin master
git push origin "$tag"
```

This triggers `.github/workflows/release.yml`, which:
- verifies the tagged commit is reachable from `master`
- builds artifacts once
- runs unit and install-smoke validation
- publishes the pre-release to TestPyPI only
- creates or updates the GitHub Release as a pre-release

NOTE: if this fails, usually need to re-tag and push.


### stable release to PyPI
Use a stable tag only after the release commit is already pushed to `origin/master`.

```bash
# 1) start from an up-to-date master branch
git checkout master
git pull --ff-only origin master

# 2) check existing tags and decide on yours. stable tags look like v0.1.3
git tag --sort=-v:refname | grep '^v' | head -n4

# 3) create and push an annotated pre-release tag
tag="v0.1.4"
git tag -a $tag -m "Release $tag"
git push origin master
git push origin "$tag"
```

This triggers `.github/workflows/release.yml`, which:
- verifies the tagged commit is reachable from `master`
- builds artifacts once
- runs unit and install-smoke validation
- publishes the stable release to PyPI only
- creates or updates the GitHub Release from the same tag

 

## validating the trigger

After pushing a tag:
- Open [GitHub Actions](https://github.com/cefect/floodsr/actions) and confirm the `Release` workflow started from the tag.
- Check the [GitHub Releases](https://github.com/cefect/floodsr/releases) page for that same tag.
- Check [PyPI](https://pypi.org/project/floodsr/) or [TestPyPI](https://test.pypi.org/project/floodsr/) for the new release and verify the version matches the tag.

## quick post-publish containerized checks
NOTE: wont work from inside devcontainer

### TestPyPI

```bash
# `--rm` removes the container after exit; `--init` helps reap child processes cleanly.
docker run --rm --init condaforge/miniforge3:25.3.1-0 bash -lc "
  set -euo pipefail &&
  export PIPX_HOME=/opt/pipx &&
  export PIPX_BIN_DIR=/usr/local/bin &&
  python -m pip install --upgrade pip pipx &&
  pipx install --index-url https://test.pypi.org/simple/ --pip-args='--extra-index-url https://pypi.org/simple' 'floodsr==${tag#v}' &&
  pipx runpip floodsr show floodsr &&
  floodsr doctor &&
  floodsr models list
"
```



```bash
# extended
docker run --rm --init condaforge/miniforge3:25.3.1-0 bash -lc "
  set -euo pipefail &&
  conda create -n floodsr-gdal -c conda-forge python=3.12 gdal pcraster -y &&
  conda run -n floodsr-gdal python -m pip install --upgrade pip &&
  conda run -n floodsr-gdal python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple floodsr==${tag#v} &&
  conda run -n floodsr-gdal python -m pip show floodsr &&
  conda run -n floodsr-gdal floodsr doctor &&
  conda run -n floodsr-gdal floodsr models list
"
```

### PyPI

```bash
docker run --rm --init condaforge/miniforge3:25.3.1-0 bash -lc "
  set -euo pipefail &&
  export PIPX_HOME=/opt/pipx &&
  export PIPX_BIN_DIR=/usr/local/bin &&
  python -m pip install --upgrade pip pipx &&
  python -m pip index versions floodsr &&
  pipx install floodsr &&
  pipx runpip floodsr show floodsr &&
  floodsr doctor &&
  floodsr models list
"
```





## setup (one time)

### local packaging tools

Use the .devcontainer `dev` image. 

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
