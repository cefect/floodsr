# ADR-0014: Development Environment

This document captures the operational development environment used by contributors.

Use [`CONTRIBUTING.md`](/workspace/CONTRIBUTING.md) for the short contributor workflow and setup entrypoint.
Use [`docs/dev/adr/`](/workspace/docs/dev/adr) for durable architectural decisions.

## Current split

- Use `.devcontainer/main` for source development, code execution, and tests.
- Use the `deploy` conda environment inside `.devcontainer/main` for code and test work.
- Use `.devcontainer/docs` for Sphinx documentation work.
- Keep development outside the containers limited to simple host-side tasks unless there is a clear reason otherwise.

## When to update the devcontainer

Update the devcontainer when the development image no longer matches the supported development workflow. Typical triggers include:

- the runtime or development dependency set changes in a way that affects imports, tests, or CLI behavior
- the `deploy` or `dev` conda environments change materially
- VS Code/devcontainer settings need to change for the supported workflow
- authentication, mounts, cache paths, or other required container wiring changes
- the base image, Dockerfile stages, or image tags are intentionally refreshed

Do not rebuild or retag the devcontainer for incidental repo changes that do not affect the development environment.

## How to update the devcontainer

The detailed build and export steps live in [`container/miniforge/readme.md`](/workspace/container/miniforge/readme.md).

Summary:

- build the relevant image target from `container/miniforge/Dockerfile`
- refresh the exported lockfiles and package snapshots as needed
- update `.devcontainer/main/docker-compose.yml` to point at the new dev image tag
- validate the container with a small set of representative imports and tests
- push the new dev image tag to the registry
- update [`CONTRIBUTING.md`](/workspace/CONTRIBUTING.md) if the setup steps, image tag, required env vars, or expected workflow changed

## Maintenance expectations

- The dev image tag referenced by `.devcontainer/main/docker-compose.yml` should point to a pushed image, not just a local build.
- Contributor-facing setup guidance should stay aligned with the currently supported dev image and container workflow.
- If a change is architectural rather than operational, capture it in an ADR instead of expanding this document.
