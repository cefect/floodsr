# Plan

Architecture decisions are tracked in ADRs: `docs/dev/adr`

 
## Delivery Plan  

### Phase 1 - Define inference contract

- [x] lock ONNX model I/O: tensor names, channel order, dtype, nodata handling,  output scaling/clipping. see  `docs/dev/adr/0001-mvp-runtime-and-cli-contract.md` 
- [ ] add a small golden test case

### Phase 2 - ONNX Runtime (ORT) CPU engine MVP
ADR refs: `docs/dev/adr/0001-mvp-runtime-and-cli-contract.md`, `docs/dev/adr/0004-repository-and-module-architecture.md`

- [x] implement `EngineORT` session load/run path
- [x] add deterministic CPU unit tests

### Phase 3 - CLI MVP
ADR refs: `docs/dev/adr/0001-mvp-runtime-and-cli-contract.md`, `docs/dev/adr/0003-logging-policy-cli-and-library.md`, `docs/dev/adr/0004-repository-and-module-architecture.md`

- [x] implement `floodsr infer` from GeoTiff
- [x] add `floodsr doctor` backend/extras diagnostics

### Phase 4 - Model registry UX
ADR refs: `docs/dev/adr/0005-model-registry-and-artifact-validation.md`, `docs/dev/adr/0004-repository-and-module-architecture.md`

- [x] implement manifest, cache download, sha256 validation
- [x] provide `models fetch/list` commands
- [x] add integration checks for corrupted downloads

### Phase 5 - GeoTIFF I/O + tiling + stitching
ADR refs: `docs/dev/adr/0004-repository-and-module-architecture.md`

- [ ] windowed reads, tile batching, overlap/blend stitching
- [ ] preserve georeferencing metadata

### Phase 6 - Packaging and install story
ADR refs: `docs/dev/adr/0002-packaging-and-installation-strategy.md`

- [ ] CPU extras first, CUDA deferred
- [ ] document `pipx` and fallback `venv` install paths

### Phase 7 - Docs and release
ADR refs: `docs/dev/adr/0002-packaging-and-installation-strategy.md`, `docs/dev/adr/0005-model-registry-and-artifact-validation.md`

- [ ] publish install/usage/cache-location/offline-use docs

### Phase 8 - GPU extension planning (deferred)
ADR refs: `docs/dev/adr/0001-mvp-runtime-and-cli-contract.md`, `docs/dev/adr/0002-packaging-and-installation-strategy.md`

- [ ] keep hooks for future provider-specific extensions

 
