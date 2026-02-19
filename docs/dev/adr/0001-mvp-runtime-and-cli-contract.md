# ADR-0001: MVP Runtime and CLI Contract
 

The project needs an MVP inference path that works cross-platform and can be called both from shell scripts and a QGIS plugin subprocess.

## Decision

- Artifact format: ONNX model files.
- Runtime for MVP: ONNX Runtime (CPU).
- User-facing endpoint: CLI.
- Cross-platform target: Windows and Unix.
- GPU support is deferred, but architecture must keep a stable CLI contract while allowing future execution provider swaps (CUDA on Linux, DirectML on Windows, etc.).

## Consequences

- The inference engine implementation can evolve without breaking CLI callers.
- CPU path is prioritized for reliability and packaging simplicity.
- Future GPU support should be additive, not a rewrite.

