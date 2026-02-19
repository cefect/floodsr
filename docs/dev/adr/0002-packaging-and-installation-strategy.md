# ADR-0002: Packaging and Installation Strategy

 
End users need a clean install path that does not pollute host environments (including QGIS-managed Python), while allowing CPU/GPU runtime variants.

## Decision

- Ship one pip package providing:
  - Python library: `floodsr`
  - CLI entrypoint: `floodsr`
- Recommend `pipx` as the primary installation method for end users.
- Keep ONNX Runtime as optional extras:
  - `floodsr[cpu]` -> `onnxruntime`
  - `floodsr[cuda]` -> `onnxruntime-gpu` (deferred/optional)

## Consequences

- Users get isolated installs by default.
- Runtime variants are explicit and conflict-aware (`onnxruntime` vs `onnxruntime-gpu`).
- Packaging remains flexible as GPU support expands.

