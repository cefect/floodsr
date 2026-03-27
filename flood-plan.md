# FloodSR Plan Of Action

## Currently, we only have the one inference onnx weights based downscaling model.

# OBJECTIVE

Create a full plan to implement the CostGrow terrain penalty method from the project /Users/walter/FloodDownscaler2 

The plan should break down the tasks into manageable and reviewable worksize chunks. 

A branch has been created to implement this work. 

# context

* CostGrow terrain-penalty is rules-based
* loosely based on my article: <https://hess.copernicus.org/articles/28/575/2024/>
* FloodDownscaler2 uses whitebox tool for the cost penalty algos... which is not well suited for our architecture (without nasty external dependencies)
* I created a standalone notebook version that uses PCRaster instead of whitebox tools [here](https://github.com/cefect/FloodDownscaler2/blob/main/misc/example_pcraster.ipynb).
* I ported this to our conda -n dev environment and it seems to be working [here](https://github.com/cefect/floodsr/commit/257d3f823b1a1a9d68559fcc335d9b2c55823d5c)

# task

* implement the new **CostGrow-Terrain** model per the ADRs. The first work will go into costgrow-terrain-penalty
* probably needs a new provider?
* will need to update the .toml for the new dependency and re-prove the installs (let me know when you're ready and I can start my local runner for '.github/workflows/install-edge.yml'). report on the change in install time with the new heavier footprint. devcontainer shouldn't need changing
* Existing architecture is supposed to support additional models like this... but it is unproven... so you'll probably need to massage some things. but minimize changes to the ResUNet\_16x\_DEM path and try and stick within the architecture... spend some time to think carefully about how it all should fit together.
* ensure we maintain the same non-window and windows (gdal) flow paths (prove large rasters won't blow up memory)
* add tests (esp extending the regression tests)
* add docs
* add a tutorial for comparing the two models
* as this is a big job, We will  want use the branch 'dev\_cg' and feature branch off of that.
* 

---

# Implementation Plan

## Architecture Overview (before starting)

### How the current system works

```
CLI (cli.py)
  └── tohr() [tohr.py]
        ├── resolve_model_worker_class()  → loads models/<ModelClass>.py dynamically
        ├── preprocessing (CRS, resample)
        └── Model.run(depth_lr_fp, dem_hr_fp, ...) → output raster
              └── (ResUNet only) EngineORT.run_tile() → ONNX inference
```

The `model_registry.py` manifest (`models.json`) maps `model_version` strings to downloadable ONNX weight files. The `Model` base class (`models/base.py`) wraps both the file path and the engine.

### Key design decisions for CostGrow

1. **No weights file**: CostGrow is rules-based and has no ONNX weights artifact. It should be treated as a built-in worker method, not a downloadable model.

2. **No Engine needed**: CostGrow does not use `EngineBase` / `EngineORT`. The model worker does all computation itself using PCRaster and numpy, so `engine/` should remain untouched unless a small runtime probe helper is genuinely useful.

3. **PCRaster as a core conda dependency**: PCRaster will be added to the core conda environment spec (`conda-env-deploy.yml` and lock files). Since pcraster is conda-forge only (not pip-installable), keep a lazy import guard as a belt-and-suspenders fallback for pip-only environments, but all standard supported environments will ship with PCRaster.

4. **Basic `pip` / `pipx` functionality must remain intact**: Existing basic installs, existing tutorials, and the ResUNet path must continue to work with plain `pip install floodsr` / `pipx install floodsr`.

5. **Tutorial should stay close to the existing pip-first flow**: Prefer live CostGrow execution in the tutorial if Colab can be patched cleanly to install PCRaster alongside the current GDAL setup. If that turns out to be awkward or brittle, fall back to precomputed CostGrow outputs and document the limitation plainly.

6. **Consistent windowed I/O with ResUNet**: CostGrow should follow the same simple-path vs. large-raster windowed/VRT flow as ResUNet so the architecture stays consistent. The distinction is that CostGrow still performs a global solve; the windowing machinery is for raster materialization and memory control, not independent tile-wise inference.

7. **Minimal disruption to ResUNet path**: All changes to shared code (`tohr.py`, `cli.py`, `model_registry.py`, `models/base.py`) must be backward-compatible. ResUNet tests must continue to pass unchanged.

---

## Branch Strategy

```
main
 └── dev_cg            ← integration branch (PR'd into main when all features land)
       ├── feat/cg-01-deps
       ├── feat/cg-02-arch
       ├── feat/cg-03-implementation
       ├── feat/cg-04-docs
```

Each `feat/cg-*` branch is PR'd into `dev_cg` (not `main`) for review.

---

## PR Breakdown

---

### PR 1 — `feat/cg-01-deps`: PCRaster runtime gating + install proof alignment

**Goal**: Add clear runtime gating and install proof for PCRaster without breaking the ADR-defined basic install contract.

**Scope**:

- Add `pcraster` to the core conda environment spec (`conda-env-deploy.yml` and lock files). Since pcraster is conda-forge only, this is a conda dependency — not a `pyproject.toml` entry.
- Leave `pyproject.toml` unchanged unless a small compatibility shim is genuinely required.
- `.github/workflows/install-edge.yml`: extend the existing **extended** install proof with a CostGrow-capable case and a simple `import pcraster` smoke test. Report install time delta vs. the current extended baseline.
- **Do not** change `devcontainer` (as noted in task requirements).
- Add a `_check_pcraster()` helper in a new `floodsr/engine/pcraster_check.py` (or inline in the model worker) that raises a clear `ImportError` if PCRaster is missing at runtime (belt-and-suspenders for pip-only users).

**Acceptance criteria**:

- `pcraster` is present in the standard conda environment after `conda env update`.
- In the supported conda environment, `import pcraster` succeeds.
- `python -m pip install floodsr` still works in a pip-only environment (no hard import-time failure).
- Install proof reports the incremental time/cost of adding PCRaster to the conda environment.
- `fast` test suite still passes with no changes.

**Review size**: Small — conda env / workflow / runtime-guard changes.

---

### PR 2 — `feat/cg-02-arch`: Architecture extension for rules-based models

**Goal**: Make the CLI, `tohr.py`, and model worker contract cleanly support built-in rules-based methods that have no downloadable weights file, without breaking ResUNet.

**Scope**:

**`cli.py`**:

- Extend `_resolve_tohr_model_spec()` to support built-in workers that do not require a local artifact path.
- Keep the existing `--model-version` surface as the primary selector. Do not introduce a parallel `--method` concept unless there is a strong reason.
- Ensure default behavior remains unchanged for ResUNet users.

**`model_registry.py`**:

- Start with the minimal-change approach: shim `CostGrow_Terrain` into the existing model listing/resolution flow with placeholder manifest values if needed.
- Only introduce a separate built-in registry path if the shim becomes awkward or causes too much special-case logic.
- `models list` should include both downloadable manifest models and built-in methods with a clear annotation.

**`tohr.py`**:

- Accept `model_fp=None` for built-in workers.
- Continue resolving the worker class in the usual way and pass a nullable model artifact through to the worker constructor.

**`models/base.py`**:

- Update the base contract so `model_fp` may be `None` for rules-based workers.
- Add an explicit class-level capability flag such as `requires_model_artifact: bool = True`.

**Acceptance criteria**:

- `resolve_model_worker_class("CostGrow_Terrain")` returns the correct class once the worker exists.
- CLI model resolution can return `(model_version, None)` for built-in methods.
- ResUNet path: no behavioral change, all existing tests pass.
- New unit tests confirm built-in models are listable and resolve without requiring artifact download.

**Review size**: Small-Medium — touches shared plumbing but only adds, doesn't rewrite.

---

### PR 3 — `feat/cg-03-implementation`: CostGrow implementation

**Goal**: Deliver the full runnable CostGrow implementation in one reviewable PR: core algorithm, large-raster/windowed behavior, CLI/tohr wiring, and regression/integration tests.

**New file**: `floodsr/models/CostGrow_Terrain.py`

Port the algorithm from the FloodDownscaler2 proof-of-concept notebook and `fdsc/alg/costGrow.py`, keeping the implementation here tightly aligned with that reference rather than re-documenting the internal step list in this plan.

**Class structure**:

```python
class CostGrowTerrain(Model):
    model_version = "CostGrow_Terrain"
    requires_weights = False
    windowed_io_min_bytes = <TBD based on memory profiling>

    def __init__(self, model_fp=None, **kwargs):
        ...  # model_fp ignored

    def run(self, depth_lr_fp, dem_hr_fp, output_fp, **kwargs) -> dict:
        ...

    @classmethod
    def is_valid(cls, model_fp) -> bool:
        return True  # no weights file needed
```

**Key implementation notes**:

- Use existing `preprocessing.py` utilities for resampling where possible.
- Patch the CostGrow path to match the current FloodSR interface: start from low-resolution flood depths, convert to WSE during preprocessing, run CostGrow in WSE space, and return the expected output without widening the CLI contract yet. Track the deeper depth/WSE input cleanup separately under [issue #44](https://github.com/cefect/floodsr/issues/44).
- Wrap `import pcraster` in a lazy import with the `_check_pcraster()` helper from PR 1.
- Return a metadata dict consistent with ResUNet: `{"runtime_s": ..., "method": "CostGrow_Terrain", ...}`.
- Keep the current `depth_lr_fp` / `--in` naming for now even though CostGrow internally operates on WSE.

**Acceptance criteria**:

- `CostGrowTerrain.run()` produces output matching the FloodDownscaler2 reference notebook on toy data (within floating-point tolerance).
- `fast` unit tests cover: cost surface computation, source ID map construction, lookup table mapping, NaN masking.
- No pcraster import at module load time (lazy import only).
- ResUNet tests unchanged.

**Review size**: Medium-Large — main feature implementation.

---

### PR 3A — Large-raster memory handling (within `feat/cg-03-implementation`)

**Goal**: Reduce avoidable memory pressure for CostGrow on large rasters and document what is, and is not, guaranteed for a global solve.

**Scope**:

Use the same windowing methods as ResUNet for large rasters, specifically the GDAL VRT/materialization path already used in the project. CostGrow still remains a global solver, so the consistency target here is workflow and memory handling, not making it tile-native.

**Implementation**:

- Profile memory usage of each step in PR 3 on synthetic large rasters (e.g. 10k×10k cells).
- Reuse the same `windowed_io_min_bytes` threshold as ResUNet initially (currently expected to be around the existing 16 GB setting), then only revisit if profiling shows a clear need.
- Reduce unnecessary full-array duplication before and after the PCRaster spread step while preserving the current simple-path/no-windowing behavior for smaller rasters.
- Document clearly that the threshold selects raster materialization strategy only, not independent tile-wise CostGrow inference.
- Define an explicit tested upper bound for supported raster sizes in the extended environment if full "won't OOM" proof is not realistic across all platforms.
- Document platform limitations honestly, especially where PCRaster support is weak.

**Acceptance criteria**:

- Synthetic large-raster profiling demonstrates reduced peak memory relative to the naive implementation.
- Memory profile documented in PR description.
- Same output as non-windowed path on small rasters.

This work stays in `feat/cg-03-implementation`; it is not a separate PR.

---

### PR 3B — CLI and `tohr` pipeline integration (within `feat/cg-03-implementation`)

**Goal**: Wire CostGrow into the CLI and `tohr()` entry point so users can invoke it the same way they use ResUNet.

**Scope**:

**`cli.py`**:

- Use `--model-version` to select between `ResUNet_16x_DEM` and `CostGrow_Terrain`.
- Default: existing behavior (ResUNet), so this is purely additive.
- Help text: describe CostGrow as "rules-based terrain penalty, built-in worker, requires PCRaster in an extended environment".
- `floodsr models list` output: include `CostGrow_Terrain` with `(built-in, no download)` annotation.

**`tohr.py`**:

- Use the arch changes from PR 2 to skip download for CostGrow.
- Pass through any CostGrow-specific parameters (if any) via `**kwargs`.

**Acceptance criteria**:

- `floodsr tohr --model-version CostGrow_Terrain --in ... --dem ... --out ...` works end-to-end on test data in a supported extended environment.
- `floodsr models list` shows CostGrow.
- `floodsr doctor` works without crashing when pcraster is absent.
- Existing CLI tests (`test_cli_tohr.py`, `test_cli_models.py`) still pass.

This work stays in `feat/cg-03-implementation`; it is not a separate PR.

---

### PR 3C — Regression and integration tests (within `feat/cg-03-implementation`)

**Goal**: Extend the test suite to cover CostGrow end-to-end, including a regression test that catches algorithm drift.

**Scope**:

**New test files**:

- `tests/test_costgrow_unit.py` (`fast` marker):
  
  - Cost surface computation correctness (against hand-crafted numpy arrays).
  - Source ID map: wet/dry cell assignment.
  - WSE lookup table: round-trip correctness.
  - NaN masking edge cases (all cells dry, all cells wet, isolated wet cells).
  - `CostGrowTerrain.is_valid()` always returns True.
  - Import guard: pcraster unavailable raises `ImportError` with helpful message.

- Extend the existing data-driven regression structure in `tests/test_tohr_regression.py` rather than introducing a separate parallel regression framework.

- If a separate `tests/test_costgrow_regression.py` is still useful, keep it narrowly focused on one synthetic parity case and treat it as supplemental, not the primary regression contract.

- `tests/test_cli_costgrow.py` (`fast` + CLI mock):
  
  - CLI `tohr --model-version CostGrow_Terrain` argument parsing.
  - `models list` includes CostGrow entry.
  - Missing pcraster: CLI exits with a user-friendly error.

**Extend existing**:

- `tests/test_model_registry.py`: confirm CostGrow is listable, not downloadable.
- `tests/test_tohr_regression.py`: use the existing committed regression data and metrics structure rather than adding new raster fixtures.

**Acceptance criteria**:

- `pytest -m fast` passes with no new failures.
- `pytest -m "e2e and not local"` passes in CI using the existing committed regression/tutorial data where applicable.
- Coverage does not drop below current baseline.

This work stays in `feat/cg-03-implementation`; it is not a separate PR.

---

## Sequencing & Dependencies

```
PR 1 (deps)
  └── PR 2 (arch)        ← can start alongside PR 1
        └── PR 3 (implementation: core + memory + cli + tests)
              └── PR 4 (docs)   ← depends on all above
```

PRs 1 and 2 can be developed in parallel. PR 3 starts once the minimal dependency/runtime and architecture choices are in place. PR 4 follows once the implementation and tests have landed in `dev_cg`.

---

## Resolved Direction

1. **Registry approach for built-in models**: Start with the minimal-change path by shimming `CostGrow_Terrain` into the existing model listing/lookup flow with dummy manifest values if needed. Reassess only if that becomes awkward or leaks too much special-case logic.

2. **Input naming convention / WSE conversion**: Resolved for this work: keep the current FloodSR depth-oriented interface, convert depth to WSE during preprocessing for the CostGrow path, and defer broader dual depth/WSE API support to [issue #44](https://github.com/cefect/floodsr/issues/44).

3. **`windowed_io_min_bytes` threshold for CostGrow**: Start with the same threshold currently used by ResUNet (expected to be the existing 16 GB value) and validate it against the existing Tutorial 3-sized data during PR 4.

4. **Regression data scope**: Resolved for this work: do not add new committed test rasters. Reuse the data already in the repository and store CostGrow regression expectations in the existing `case_spec.json` flow.

5. **PCRaster on Windows / Colab**: Windows is expected to be fine. For Colab, try patching the prebuilt environment similarly to the current GDAL setup; if it works cleanly, keep live execution, and if not, fall back to precomputed tutorial outputs without blocking the feature.

---

## PR 4 — `feat/cg-04-docs`: Docs and comparison tutorial

**Goal**: Document the new model, its install/runtime requirements, and how it compares with ResUNet in practice.

**Scope**:

- Add user docs for `CostGrow_Terrain`, including when to use it, current limitations, and the depth-to-WSE preprocessing note.
- Update install docs for the extended environment / PCRaster path, including any Colab-specific setup if that proves workable.
- Add or extend the comparison tutorial so users can compare ResUNet and CostGrow on existing committed tutorial data.
- Keep the tutorial aligned with the current tutorial flow: prefer live execution if PCRaster installation is reliable there; otherwise ship precomputed CostGrow outputs and explain why.

**Acceptance criteria**:

- Docs clearly explain the model selection tradeoffs and runtime requirements.
- Tutorial demonstrates both models on existing committed data.
- Colab/basic tutorial behavior is documented accurately based on what actually works.

**Review size**: Small-Medium — docs and tutorial only.

---

## Success Definition

The `dev_cg` branch is ready to PR into `main` when:

- [ ] All 4 feature PRs have been merged into `dev_cg`.
- [ ] `pytest -m "fast and not local"` passes in CI on `dev_cg`.
- [ ] `pytest -m "e2e and not local"` passes in CI on `dev_cg` (or a suitable subset).
- [ ] `floodsr tohr --model-version CostGrow_Terrain` works end-to-end in a supported extended environment.
- [ ] `floodsr tohr` with the default/model-version ResUNet path still works identically to pre-CostGrow.
- [ ] Basic `pip` / `pipx` install paths still work unchanged.
- [ ] Install time delta from adding PCRaster to the extended environment is documented in the `dev_cg` PR description.
- [ ] The comparison tutorial notebook follows the existing tutorial flow and is documented according to what actually works: live CostGrow execution where PCRaster install is reliable, otherwise precomputed CostGrow outputs.
