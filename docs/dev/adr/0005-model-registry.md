# ADR-0005: Model Registry and Artifact Validation

 

Model artifacts need versioned discovery and integrity guarantees prior to inference.

## Decision

- Maintain a `models.json` manifest mapping `version -> {url, sha256, metadata}`.
- Store fetched artifacts in the shared FloodSR cache policy defined in `ADR-0012`.
- Enforce checksum validation (`sha256`) before model use.
- Provide CLI commands:
  - `floodsr models list`
  - `floodsr models fetch <version>`
- Test requirement:
  - Corrupt or mismatched downloads must fail checksum validation.

Current reference model:

-  16x DEM-conditioned ResUNet [4690176_0_1770580046_train_base_16]
  - ONNX inference artifact from release `v2026.02.19`
  - release may include extra files not consumed by inference
  - repository/access may be private

see `docs/dev/adr/0014-model-io.md`

## Consequences

- Model resolution is explicit and auditable.
- Integrity checks prevent silent model corruption.
- CLI supports offline reuse via cached artifacts managed by the shared cache policy.
