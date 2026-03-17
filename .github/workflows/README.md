# GitHub Workflows

This directory contains the active GitHub Actions workflows for this repository.

## See Also

- See [`docs/dev/adr/0017-cicd-workflow-policy.md`](../../docs/dev/adr/0017-cicd-workflow-policy.md) for the CI/CD policy that defines the intent behind these workflows.

## Running

Push the branch first, then run a workflow from the GitHub UI or CLI.

```bash
gh workflow run --ref "$(git branch --show-current)"
 
```

## `ci.yml`

- Purpose: branch CI for pull requests and pushes to `master`.
- Scope: runs the `fast` pytest tier, runs one constrained minimum-core test slice, builds `dist/*`, runs `twine check`, and smoke-tests the core wheel on Ubuntu and Windows.
- Trigger: `pull_request`, `push` to `master`, `workflow_dispatch`.

## `install-edge.yml`

- Purpose: manual install-matrix validation without slowing normal CI.
- Scope: builds `dist/*` on the self-hosted Linux runner fleet, then runs each install smoke case in a fresh `condaforge/miniforge3:25.3.1-0` Docker container on `CEFTOP25M`; the matrix covers Linux-only basic `pipx` host-GDAL contexts plus one extended conda-forge GDAL case.
- Trigger: None (i.e., `workflow_dispatch`).

## `release.yml`

- Purpose: tagged release validation and publish workflow.
- Scope: verifies tag ancestry, runs the fast suite, runs one constrained minimum-core test slice, builds and validates `dist/*`, smoke-tests the core install on Ubuntu, smoke-tests the extended conda install on Ubuntu and Windows, then publishes to TestPyPI or PyPI and updates the GitHub Release.
- Trigger: `push` tags matching `v*`.
 
