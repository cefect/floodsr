# Release and Publishing Strategy Using Trusted Publishing

 
## Context

The project is distributed as a Python package via PyPI and installed primarily via `pipx` for local CLI use and `pip` for notebook-first environments such as Google Colab. We require a repeatable, low-risk release process that minimizes credential exposure and ensures packaging correctness before publishing stable releases. TestPyPI is available as a staging index. GitHub Actions is used for CI/CD.

Key constraints:

* Avoid long-lived API tokens in CI.
* Ensure releases are reproducible from immutable source.
* Prevent accidental publication of unstable builds to PyPI.
* Keep the workflow simple and maintainable.
* Keep Git tags, GitHub Releases, and published package versions synchronized.

## Decision

1. Use Git tags as the sole release trigger.

   * Tags matching `vX.Y.ZrcN`, `vX.Y.ZaN`, or `vX.Y.ZbN` are treated as pre-releases.
   * Tags matching `vX.Y.Z` are treated as stable releases.
   * Tags are evaluated by release workflows after verifying the tagged commit is reachable from `master`.

2. Use `setuptools-scm` so package versions are derived from Git tags rather than a static `[project].version`.

   * Release tags remain the source of truth for published versions.
   * This keeps GitHub Release tags and published package versions synchronized.
   * For trusted release/version semantics, only versions derived from commits reachable from `master` should be treated as authoritative release identifiers.
   * Versions observed on `dev`, feature branches, or stale editable installs are useful diagnostics, but they should not be treated as the trusted project release version.

3. Use PyPI Trusted Publishing (OIDC) for both TestPyPI and PyPI.

   * No API tokens stored in repository secrets.
   * Each index is configured separately with a Trusted Publisher entry.

4. Publishing policy:

   * Pre-release tags publish to TestPyPI only.
   * Stable tags publish to PyPI only.
   * Stable releases are not duplicated to TestPyPI.
   * The GitHub Release is created or updated from the same tag used for package publishing.

5. Runtime diagnostics and CLI version reporting:

   * `floodsr --version` should report the installed package version available in the active Python environment.
   * `floodsr doctor` should report the same installed package version and may also report the loaded package/module path to help debug stale editable installs.
   * When interpreting a reported version for release trust, only a version tied to `master` release ancestry is authoritative; diagnostic output from other branches remains informational only.

6. See `ADR-0017` for CI/CD workflow policy.

## CI/CD summary

See `ADR-0017` for CI/CD workflow policy.

 
