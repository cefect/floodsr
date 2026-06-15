# GitHub Workflows

This directory contains the active GitHub Actions workflows for this repository.

## See Also

- See [`docs/dev/adr/0017-cicd-workflow-policy.md`](../../docs/dev/adr/0017-cicd-workflow-policy.md) for the CI/CD policy that defines the intent behind these workflows.

## Running

Push the branch first, then run a workflow from the GitHub UI or CLI.

```bash
# launch teh local runner (needed for AllTests and Instal Edge?) [wsl only]
/home/cefect/LS/09_REPOS/04_TOOLS/gh_runner/actions-runner/run.sh

# push changes (MUST BE PUSHED!)
git push

# select the workflow to run:
# WARNING! make sure you dont have too many containers running
gh workflow run --ref "$(git branch --show-current)"

# OR... dispatch all manually runnable workflows except release
for workflow in ci.yml install-edge.yml all-tests.yml; do
    gh workflow run "$workflow" --ref "$(git branch --show-current)"
done

# cancel non-completed running workflows
gh run list --branch "$(git branch --show-current)" --limit 100 --json databaseId,status \
  --jq '.[] | select(.status != "completed") | .databaseId' | xargs -r -n1 gh run cancel

 
```

see the [tests readme](../../tests/README.md) also

## `ci.yml`

- Purpose: branch CI for pull requests and pushes to `master`.
- Scope:
  - runs the `fast` pytest tier
  - runs one constrained minimum-core test slice
  - builds `dist/*`
  - runs `twine check`
  - smoke-tests the built core wheel on Ubuntu and Windows
  - smoke-tests the extended conda install on Ubuntu and Windows
- Trigger: `pull_request`, `push` to `master`, `workflow_dispatch`.

## `install-edge.yml`

- Purpose: manual install-matrix validation without slowing normal CI.
- Scope:
  - builds `dist/*` on the self-hosted Linux runner fleet
  - proves each documented Unix install path from `docs/user/installation.rst` in a fresh context-specific Docker container
  - uses sparse-checkout notebook shims from `.github/workflows/artifacts` for notebook cases
  - uses `cefect/floodsr:install-edge-main-v0.2` for non-Colab paths
  - uses `cefect/floodsr:install-edge-colab-v0.2` for Colab paths
- See also:
  - [`docs/user/notebooks/readme.md`](../../docs/user/notebooks/readme.md) for how this install proof relates to tutorial notebook proofing
- Trigger: None (i.e., `workflow_dispatch`).

## `all-tests.yml`

- Purpose: manual full-suite validation on the self-hosted Linux runner fleet without changing branch CI scope.
- Scope:
  - runs one self-hosted job that recreates the locked `deploy` conda environment and runs `pytest -m "not sphinx and not notebook and not local"`
  - runs one self-hosted job that starts from the same locked deploy environment, layers on notebook runtime packages, and runs `pytest -m "notebook" tests/test_tutorials.py`
  - preserves logs and JUnit XML for both jobs
- See also:
  - [`docs/user/notebooks/readme.md`](../../docs/user/notebooks/readme.md) for where the real tutorial notebooks are proved
- Trigger: None (i.e., `workflow_dispatch`).

## `release.yml`

- Purpose: tagged release validation and publish workflow.
- Scope:
  - verifies tag ancestry from `master`
  - runs the fast suite
  - runs one constrained minimum-core test slice
  - builds and validates `dist/*`
  - smoke-tests the core install on Ubuntu
  - smoke-tests the extended conda install on Ubuntu
  - publishes to TestPyPI or PyPI
  - updates the GitHub Release
- Trigger: `push` tags matching `v*`.
 
