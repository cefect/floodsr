# ADR-0017: CI/CD Workflow Policy

## Context

Several ADRs define packaging, testing, and publishing intent, but the GitHub Actions workflow policy should live in one place so CI/CD topology can change without repeating workflow detail across multiple ADRs.

## Decision

- CI owns packaging validation for published artifacts.
- CI builds release artifacts once and reuses them across downstream workflow steps.
- GitHub Actions test selection is limited to `fast` tests and excludes `local`.
- Branch CI and release publishing are separate workflows with separate privileges:
    - `ci.yaml` for branch CI and PR validation.
    - `release.yaml` for tag-triggered release publishing. 
        - Trusted Publishing
        - checks out with full history, verifies the tagged commit is reachable from `master`, builds artifacts once, runs validation/tests, publishes, and then creates or updates the GitHub Release for that tag.
        - see release semantics in `ADR-0013`.
        - Release versions are derived from Git tags via `setuptools-scm`.
        - Publish jobs receive `id-token: write` only at the publish step/job boundary; build and test jobs remain read-only.
- `.github/workflows/install-edge.yml` owns Unix install-path proof coverage for the install guidance documented in `docs/user/installation.rst`.
- `install-edge.yml` should run one isolated matrix case per documented Unix install path using an appropriate Docker image for that user-facing context.
- On Windows, only the documented `basic` CLI install path must be proven.

## Implementation Notes

- Prefer `push.tags` for release triggers rather than `workflow_run` chaining for publishing.
- Reuse artifacts within the same workflow run via `upload-artifact` and `download-artifact`.
- Manual edge-install validation may run on the repository self-hosted Linux runner fleet and use a fresh job container per matrix case for reproducible install smoke coverage.
- Container selection for `install-edge.yml` should favor the closest reasonable match to the documented user environment for each Unix path rather than one generic image for all cases.
- Use GitHub environments for `testpypi` and `pypi` publish jobs.
 

## Cross-References

- `ADR-0002` owns packaging intent.
- `ADR-0006` owns test semantics.
- `ADR-0013` owns release semantics.
- `.github/workflows/README.md` documents the concrete workflow files and their operator-facing usage.
