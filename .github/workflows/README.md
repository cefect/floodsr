# GitHub Workflows

This directory contains the active GitHub Actions workflows for this repository.

## See Also

- See [`docs/dev/adr/0017-cicd-workflow-policy.md`](../../docs/dev/adr/0017-cicd-workflow-policy.md) for the CI/CD policy that defines the intent behind these workflows.

## Running

Push the branch first, then run a workflow from the GitHub UI or CLI.

```bash
# launch teh local runner (needed for AllTests and Instal Edge?) [wsl only?]
/home/cefect/LS/09_REPOS/04_TOOLS/gh_runner/actions-runner/run.sh

# push changes (MUST BE PUSHED!)
git push

gh workflow run --ref "$(git branch --show-current)"
 
```

## Workflow Summary

|  | `ci.yml` | `install-edge.yml` | `all-tests.yml` | `release.yml` |
|---|---|---|---|---|
| Purpose | Branch CI | Manual install validation | Manual full-suite validation | Tagged release validation + publish |
| Triggers | `pull_request`, `push` to `master`, `workflow_dispatch` | `workflow_dispatch` | `workflow_dispatch` | `push` tags matching `v*` |
| Runs-on | GitHub-hosted Ubuntu + Windows | Self-hosted Linux runner fleet | Self-hosted Linux runner fleet | GitHub-hosted Ubuntu + Windows |
| Summary | - run `fast` pytest tier<br>- run constrained minimum-core slice<br>- build `dist/*`, run `twine check`<br>- smoke-test core wheel on Ubuntu and Windows | - build `dist/*` on self-hosted Linux<br>- validate documented Unix install paths in fresh Docker contexts<br>- use sparse-checkout notebook shims from `.github/workflows/artifacts`<br>- use dedicated main vs Colab install-edge images | - recreate locked `deploy` env and run `pytest -m "not sphinx and not notebook"`<br>- layer notebook runtime packages and run notebook tests<br>- preserve logs and JUnit XML for both jobs | - verify tag ancestry<br>- run `fast` pytest tier and constrained minimum-core slice<br>- build and validate `dist/*`<br>- smoke-test core and extended installs, then publish and update release |
 
