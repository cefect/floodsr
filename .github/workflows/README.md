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
- Scope: builds `dist/*` on the self-hosted Linux runner fleet, then proves each documented Unix install path from `docs/user/installation.rst` in a fresh context-specific Docker container; notebook cases use sparse checkout shims from `.github/workflows/artifacts`, and notebook containers pull from `cefect/floodsr:install-edge-jupyter-v0.1` or `cefect/floodsr:install-edge-colab-v0.1`.
- Trigger: None (i.e., `workflow_dispatch`).

## `all-tests.yml`

- Purpose: manual full-suite validation on the self-hosted Linux runner fleet without changing branch CI scope.
- Scope: runs two self-hosted jobs: one recreates the locked `deploy` conda environment and runs `pytest -m "not sphinx and not notebook"`; the second starts from the same locked deploy environment, layers on the notebook runtime packages, and runs `pytest -m "notebook" tests/test_notebooks.py`, preserving logs and JUnit XML for both jobs.
- Trigger: None (i.e., `workflow_dispatch`).

## `release.yml`

- Purpose: tagged release validation and publish workflow.
- Scope: verifies tag ancestry, runs the fast suite, runs one constrained minimum-core test slice, builds and validates `dist/*`, smoke-tests the core install on Ubuntu, smoke-tests the extended conda install on Ubuntu and Windows, then publishes to TestPyPI or PyPI and updates the GitHub Release.
- Trigger: `push` tags matching `v*`.
 
