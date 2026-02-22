# Release and Publishing Strategy Using Trusted Publishing

 
## Context

The project is distributed as a Python package via PyPI and installed primarily via pip and pipx. We require a repeatable, low-risk release process that minimizes credential exposure and ensures packaging correctness before publishing stable releases. TestPyPI is available as a staging index. GitHub Actions is used for CI/CD.

Key constraints:

* Avoid long-lived API tokens in CI.
* Ensure releases are reproducible from immutable source.
* Prevent accidental publication of unstable builds to PyPI.
* Keep the workflow simple and maintainable.

## Decision

1. Use Git tags as the sole release trigger.

   * Tags matching `vX.Y.ZrcN`, `vX.Y.ZaN`, or `vX.Y.ZbN` are treated as pre-releases.
   * Tags matching `vX.Y.Z` are treated as stable releases.

2. Use PyPI Trusted Publishing (OIDC) for both TestPyPI and PyPI.

   * No API tokens stored in repository secrets.
   * Each index is configured separately with a Trusted Publisher entry.

3. Publishing policy:

   * Pre-release tags publish to TestPyPI only.
   * Stable tags publish to PyPI only.
   * Stable releases are not duplicated to TestPyPI.

4. The CI workflow:

   * Runs on tag push.
   * Builds sdist and wheel once.
   * Runs basic packaging validation.
   * Publishes to the appropriate index based on tag type.

 