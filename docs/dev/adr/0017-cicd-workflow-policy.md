# ADR-0017: CI/CD Workflow Policy

## Context

Several ADRs define packaging, testing, and publishing intent, but the GitHub Actions workflow policy should live in one place so CI/CD topology can change without repeating workflow detail across multiple ADRs.

## Decision

- CI owns packaging validation for published artifacts.
- CI builds release artifacts once and reuses them across downstream workflow steps.
- GitHub Actions test selection excludes `network` tests.
- `sphinx` tests are local-only and do not run in GitHub Actions.
- Tag-triggered publishing workflow topology is defined here and supports the release semantics in `ADR-0013`.

## Cross-References

- `ADR-0002` owns packaging intent.
- `ADR-0006` owns test semantics.
- `ADR-0013` owns release semantics.
