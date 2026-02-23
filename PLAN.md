# Plan

Architecture decisions are tracked in ADRs: `docs/dev/adr`

 
## Delivery Plan  

### Phase 1 - Define inference contract

- [x] lock ONNX model I/O: tensor names, channel order, dtype, nodata handling,  output scaling/clipping. see  `docs/dev/adr/0001-mvp-runtime-and-cli-contract.md` 
- [x] add a small golden test case

### ONNX Runtime (ORT) CPU engine MVP
ADR refs: `docs/dev/adr/0001-mvp-runtime-and-cli-contract.md`, `docs/dev/adr/0004-repository-and-module-architecture.md`

- [x] implement `EngineORT` session load/run path
- [x] add deterministic CPU unit tests

### CLI 

- [x] implement `floodsr infer` from GeoTiff
- [x] add `floodsr doctor` backend/extras diagnostics

###   Model registry UX
ADR refs: `docs/dev/adr/0005-model-registry-and-artifact-validation.md`, `docs/dev/adr/0004-repository-and-module-architecture.md`

- [x] implement manifest, cache download, sha256 validation
- [x] provide `models fetch/list` commands
- [x] add integration checks for corrupted downloads

###  tiling + stitching
ADR refs: `docs/dev/adr/0008-tiling.md`

- [x] windowed reads, tile batching, overlap/blend stitching
- [x] preserve georeferencing metadata

###   Packaging/install and pupublishing CI/CD
ADR refs: `docs/dev/adr/0002-packaging-and-installation-strategy.md`

- [ ] try publish to PyPi (manually). see `docs/dev/adr/0013-publishing.md`
- [ ] automate build/test/publish to PyPi  w/ trusted publishing. see `docs/dev/adr/0013-publishing.md`
- [ ] nice test coverage with fast feedback loops
- [ ] add some CI/CD badges to README
- [ ] think about depencies (what ranges should go in toml?) Ci/CD depency matrix? 



### Documentation
- [x] publish ENG to ReadTheDocs
- [ ] add tutorial. write as notebooks and port to RTD? shiould be based on tutorial data. see `examples.ipynb`. Just one tutorial for now, showing model fetching, switching, and `tohr`. i.e., cover most of CLI. 
- [ ] setup for french version
- [ ] LLM french translation
- [ ] human proof


 

 
### release v1.0.0
- [ ] update changelog
- [ ] tag and release on GitHub
- [ ] pretty picture on readme.md. maybe from `tests/data/rss_dudelange_A` with a zoom in and the lores on the left and hires on the right. using a nice coloscale and the DEM (as hillshade) as a basemap. should pop and be sexy. 

### add costgrow_pcraster model  feature
- [ ] update `container/miniforge/Dockerfile` to be more modular. rename current deploy layer to `onnx`. add a nother layer  for `pcraster` (use same syntax with environment files for each targert). add short/final layer for `deploy` to keep end point resolution the same. 
- [ ] evaluate/test how much heavier the install is now. 
- [ ] add model to registry, add `floodsr/models/costgrow_pcraster.py` and other implementation work. see `others/CostGrow_pcraster_inline.ipynb`. 
- [ ] add tests 
- [ ] documentation.

### preprocessing WSE feature
- [ ] allow ingestion of water surface rasters (with a flaga). and convert these. 

### cache behavior feature
see `docs/dev/adr/0012-cache-policy-and-lifecycle.md`
- [ ] build out modules
- [ ] update CLI
- [ ] add tests
- [ ] udpate documentation

### add building blocking feature
- [ ] identify data source: see `docs/dev/adr/0016-buildings.md`
- [ ] develop Proof of concept of fetching w/ bbox
- [ ] integrate. should be similar to dem fetching. 
- [ ] add tests
- [ ] documentation
- [ ] add to tutorial