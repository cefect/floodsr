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
        - checks out with full history, verifies the tagged commit is reachable from `main`, builds artifacts once, runs validation/tests, publishes, and then creates or updates the GitHub Release for that tag.
        - see release semantics in `ADR-0013`.
        - Release versions are derived from Git tags via `setuptools-scm`.
        - - Publish jobs receive `id-token: write` only at the publish step/job boundary; build and test jobs remain read-only.

## Implementation Notes

- Prefer `push.tags` for release triggers rather than `workflow_run` chaining for publishing.
- Reuse artifacts within the same workflow run via `upload-artifact` and `download-artifact`.
- Use GitHub environments for `testpypi` and `pypi` publish jobs.
 

## Cross-References

- `ADR-0002` owns packaging intent.
- `ADR-0006` owns test semantics.
- `ADR-0013` owns release semantics.
