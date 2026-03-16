# GitHub Workflows

This directory contains the two active GitHub Actions workflows for this repository.

## `ci.yml`

Purpose:
- Run branch CI for pull requests and pushes to `master`.
- Validate unit tests plus packaging/install smoke checks without any publish privilege.

What it does:
1. Runs `pytest -m "fast and not local and not sphinx"`.
2. Builds `dist/*` with `python -m build`.
3. Runs `twine check` on the built artifacts.
4. Smoke tests the built core wheel in isolated envs on `ubuntu-latest` and `windows-latest` without GDAL bindings.
5. Smoke tests the built extended wheel after installing system GDAL and matching Python bindings.

Triggers:
- `pull_request`
- `push` to `master`
- `workflow_dispatch`

### force run
ensure the changes are pushed. 

```bash
gh workflow run ci.yml --ref master
```

Run it against the current branch:

```bash
gh workflow run ci.yml --ref "$(git branch --show-current)"
```

Watch the most recent CI run:

```bash
gh run watch "$(gh run list --workflow ci.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
```


## `release.yml`

Purpose:
- Publish tagged releases using Trusted Publishing.
- Keep Git tags, GitHub Releases, and package versions synchronized through `setuptools-scm`.

What it does:
1. Triggers on pushed tags matching `v*`.
2. Checks out full Git history and verifies the tagged commit is reachable from `master`.
3. Builds `dist/*` once and validates that the derived package version matches the tag.
4. Runs the release validation suite and install smoke checks.
5. Publishes pre-releases to TestPyPI or stable releases to PyPI via Trusted Publishing.
6. Creates or updates the GitHub Release from the same tag.

## Running

From GitHub UI:
1. Open **Actions**.
2. Select **CI** or **Release**.
3. Inspect the run for the relevant branch or tag.

Using GitHub CLI:
```bash
gh workflow run ci.yml --ref master
```

## Interpreting failures

Typical failure buckets:
- Packaging/install errors: project metadata, dependency resolution, or wheel build failures.
- Versioning errors: the `setuptools-scm` derived version does not match the pushed tag.
- Tag policy errors: the tagged commit is not reachable from `master`.
- Trusted Publishing errors: missing or mismatched PyPI/TestPyPI publisher configuration.
- Test tier selection drift: a non-unit test leaked into CI selection.
