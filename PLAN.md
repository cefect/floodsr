## Target MVP Architecture

- **Artifact:** one or more ONNX model files (weights + graph)
- **Runtime:** ONNX Runtime (ORT) CPU for MVP
- **Endpoint:** a CLI (usable from bash/PowerShell and callable from a QGIS plugin via subprocess)
- **Cross-platform:** Windows + Unix
- **GPU later:** keep the code structured so you can swap ORT execution providers (CUDA on Linux, DirectML on Windows, etc.) without changing the CLI contract. ORT is explicitly designed around an "Execution Provider" (EP) framework for this.

## Design

Ship one pip package that provides:

- a library (`floodsr`)
- a CLI entrypoint (`floodsr`)

Recommend `pipx` as the primary installation method for end users, because it guarantees isolation and keeps QGIS clean.

Make ORT an install extra rather than a hard dependency, so you can support CPU vs GPU variants cleanly (because ORT says only one of `onnxruntime` / `onnxruntime-gpu` should be in an environment).

Concretely:

- `floodsr[cpu]` -> depends on `onnxruntime`
- `floodsr[cuda]` -> depends on `onnxruntime-gpu` (later)


## Project structure outline (repo layout)

A structure that keeps the ORT “engine” swappable and keeps CLI thin:

```

pyproject.toml
README

# CLI surface
cli.py                 # Typer app,  (download/verify/cache)
model_registry.py      # resolves model versions, URLs, sha256
cache_paths.py         # platformdirs-based paths
checksums.py           # sha256 verify

# Inference engine abstraction
engine/
    __init__.py
    base.py              # Engine interface (load, run, list_providers)
    ort.py               # ORT implementation (CPU MVP)
    providers.py         # provider selection + detection logic
    # future:
    # ort_cuda.py
 

# Data model + preprocessing
schema/
    __init__.py
    config.py            # pydantic models for CLI config
preprocess/
    __init__.py
    normalize.py         # apply stats/scale/clipping rules

# Tiling + stitching (kept runtime-agnostic)
tiling/
    __init__.py
    tiles.py             # window generation
    stitch.py            # feather/blend + assemble output

# I/O (optional geo dependency)
io/
    __init__.py
    rasterio_io.py       # GeoTIFF read/write (guarded by [geo] extra)
    npy_io.py            # lightweight fallback for arrays
    metadata.py          # CRS/transform handling

tests/
    test_registry.py
    test_engine_ort.py
    test_tiling.py
    test_cli_smoke.py
dev/                  # dev-only tools (lint/test/release helpers)
docs/
    (mkdocs.yml or sphinx config)
```

module interaction:
```bash
CLI
  ↓
model_registry.resolve_model(...)
  ↓
cache_paths.model_path(...)
  ↓
(download if needed)
  ↓
checksums.verify_sha256(...)
  ↓
return local model path
  ↓
engine.load(model_path)
```


## example CLI

* `floodsr infer --in <file> --out <file> [--model <ver|path>] [--tile N] [--overlap N]`
* `floodsr models list`
* `floodsr models fetch <version>`
* `floodsr doctor` (prints what extras are installed, what ORT providers are available)




## Work plan outline (CPU MVP first, GPU-ready architecture)

### Phase 1 — Define the inference contract  

* Lock down ONNX model I/O:

  * input tensor names, channel order, dtype, nodata handling
  * output scaling and clipping

* Write a tiny “golden” test case (small input array → expected output array).

### Phase 2 — ORT CPU engine MVP  

* Implement `EngineORT`:

  * load ONNX once (InferenceSession)
  * run one tile (NumPy in/out)
  * optional: set ORT session options + threading knobs (exposed later in CLI config)
* Add unit tests for deterministic behavior on CPU.

### Phase 3 — CLI MVP  

* Implement `floodsr infer` that can run on:

  * `.npy` inputs first (fast to develop, no geo deps),
  * then GeoTIFF once `io/rasterio_io.py` is added.

* Add `floodsr doctor` to:

  * detect whether ORT is installed (cpu/cuda/dml),
  * warn if no compatible backend is present.

### Phase 4 — Model registry + “download weights” UX  

* Implement:

  * `models.json` manifest (version → url + sha256)
  * download into OS-specific cache dir
  * sha256 verification before use
* Add `floodsr models fetch` + `floodsr models list`.
* Integration tests: corrupt download should fail checksum.

current models:

#### 4690176_0_1770580046_train_base_16
16x DEM-conditioned ResUNet. training run from the Feb training experiments. 
[release](https://github.com/cefect/floodsr/releases/tag/v2026.02.19) with:
- train_config.json: training config + stats used for normalization
- model_infer.onnx: the ONNX export of the trained model, which is what the inference code will use.

note: 
- there are more files in the release that should be ignored. 
- repo is private (for now)





### Phase 5 — GeoTIFF I/O + tiling + stitching  

 
* Implement:

  * windowed reads
  * tile batching (even on CPU this matters for overhead)
  * stitching (overlap/blend)
  * preserve georeferencing metadata

### Phase 6 — Packaging + install story  

* Implement extras: `cpu`; reserve `cuda` for later.
* Document two install paths:

  * **Preferred**: pipx install (isolated). ([Python Packaging][4])
  * **Fallback**: “create venv in chosen folder” script.

### Phase 7 — Docs + release  

* Minimal docs:

  * install (pipx / venv, Windows + Unix)
  * CLI usage examples
  * model cache location + offline use
 

### Phase 8 — GPU extension planning (deferred, but keep the hooks)

 
