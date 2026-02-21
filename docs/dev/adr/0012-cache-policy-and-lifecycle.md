# ADR-0012: Cache Policy and Lifecycle

FloodSR needs a shared cache policy that works across features, not just model weights.

## Decision

- Implement cache behavior as a dedicated cache subsystem, with CLI commands as thin adapters.
- Cache module ownership:
  - `cache/paths.py` for cache root resolution and namespace paths.
  - `cache/policy.py` for defaults and TTL configuration.
  - `cache/lifecycle.py` for expiry evaluation and purge execution.
  - `cache/reporting.py` for aggregate stats used by `cache info`.
- Use one shared FloodSR cache root for all cacheable artifacts.
- Resolve cache root in this order:
  - `FLOODSR_CACHE_DIR` if set.
  - Otherwise OS defaults:
    - Linux: `$XDG_CACHE_HOME/floodsr` or `~/.cache/floodsr`
    - macOS: `~/Library/Caches/floodsr`
    - Windows: `%LOCALAPPDATA%\\floodsr\\Cache`
- Partition cache by namespace under the root (for example: `models/`, `tiles/`, `metadata/`).
- Cache lifetime defaults to `30 days` since last access (`last_accessed_utc`).
- Expired artifacts are eligible for deletion during cache-aware operations and explicit purge commands.
- Provide user controls:
  - `floodsr cache info`
  - `floodsr cache purge`
  - `floodsr cache purge --all`
  - `floodsr cache purge --namespace <name>`
  - `floodsr cache purge --dry-run`
- `floodsr cache info` reports:
  - resolved cache directory and whether an override is active
  - artifact count and total size
  - size and count by namespace
  - configured TTL and expired artifact count

## Scope (initial)

- Current primary consumer is model weights (`models/` namespace).
- Additional features may reuse the same cache contract without redefining path, TTL, or cleanup UX.

## Consequences

- Cache behavior is consistent across the CLI and future subsystems.
- Users get clear, centralized controls for cache inspection and cleanup.
- Cache growth is bounded by default retention.
- Cache policy and lifecycle are reusable by non-CLI features without redefining behavior.
