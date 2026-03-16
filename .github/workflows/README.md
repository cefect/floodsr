# GitHub Workflows

This directory contains the active GitHub Actions workflows for this repository.

## Cross-Reference

- See [`docs/dev/adr/0017-cicd-workflow-policy.md`](../../docs/dev/adr/0017-cicd-workflow-policy.md) for the CI/CD policy that defines the intent behind these workflows.

## Running

Push the branch first, then run a workflow from the GitHub UI or CLI.

```bash
gh workflow run ci.yml --ref "$(git branch --show-current)"
gh workflow run install-edge.yml --ref "$(git branch --show-current)"
```

## `ci.yml`

- Purpose: branch CI for pull requests and pushes to `master`.
- Scope: runs the `fast` pytest tier, builds `dist/*`, runs `twine check`, and smoke-tests the core wheel on Ubuntu and Windows.
- Trigger: `pull_request`, `push` to `master`, `workflow_dispatch`.

## `install-edge.yml`

- Purpose: manual install-matrix validation without slowing normal CI.
- Scope: builds `dist/*`, tests the basic `pipx` install on Ubuntu and Windows across host-GDAL contexts, and tests the extended conda install on Ubuntu.
- Trigger: `workflow_dispatch`.

## `release.yml`

- Purpose: tagged release validation and publish workflow.
- Scope: verifies tag ancestry, runs the fast suite, builds and validates `dist/*`, smoke-tests the core install on Ubuntu, smoke-tests the extended conda install on Ubuntu and Windows, then publishes to TestPyPI or PyPI and updates the GitHub Release.
- Trigger: `push` tags matching `v*`.
 
