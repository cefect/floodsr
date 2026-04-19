# Docs Tutorials Follow-up

This document breaks the CostGrow documentation/tutorial work into small passes.
Each pass is intended to be reviewable on its own and comes with a manual test plan
so the work can be validated before moving to the next step.

## Scope

Goal:

- document the new `CostGrow_Terrain` model in the user-facing docs
- update the user guide and CLI-facing docs so the model is discoverable
- add a new tutorial 4 notebook focused on CostGrow
- reuse the structure and plotting patterns from tutorial 2
- add a quick comparison flow that runs `tohr` with both CostGrow and ResUNet
- generate a comparison raster/figure and plot the outputs side by side

Non-goals for this branch:

- changing CostGrow runtime behavior or algorithms
- changing model defaults unless the docs expose a real mismatch
- broad docs reorganization outside the pages touched by this plan

## Assumptions

- the branch base is `origin/dev_cg`
- `CostGrow_Terrain` is already implemented and runnable from `floodsr tohr`
- ResUNet remains the fast comparison baseline for the tutorial
- the tutorial should be runnable from the in-repo notebook execution pattern used by `tutorial_2.sh`
- manual validation will be performed from a local environment with the required optional dependencies installed

## Pass 0: Baseline Audit

Objective:

- confirm the current docs gaps before editing anything
- identify exact pages and notebook assets that need to change
- verify the current user-facing text does not already cover the planned material

Planned edits:

- no product-doc edits in this pass
- collect notes from:
  - `docs/user/user_guide.rst`
  - `docs/user/tutorials.rst`
  - `docs/user/cli_reference.rst`
  - `docs/user/installation.rst`
  - `docs/user/notebooks/tutorial_2.ipynb`
  - `docs/user/notebooks/tutorial_2.sh`

Manual test plan:

1. From the repo root, inspect the current user guide CostGrow section:
   - `sed -n '60,120p' docs/user/user_guide.rst`
2. Inspect the tutorials index:
   - `sed -n '1,220p' docs/user/tutorials.rst`
3. Inspect the installation page for CostGrow prerequisites:
   - `rg -n "CostGrow|PCRaster" docs/user/installation.rst`
4. Confirm there is no tutorial 4 yet:
   - `find docs/user/notebooks -maxdepth 1 -name 'tutorial_4*'`
5. Confirm tutorial 2 is the best template:
   - `sed -n '1,220p' docs/user/notebooks/tutorial_2.sh`

Expected verification results:

- `user_guide.rst` still contains the placeholder `Rules-based. Not implemented yet.`
- `tutorials.rst` only lists tutorials 1 through 3
- installation docs mention PCRaster, but there is no end-to-end CostGrow tutorial
- no `tutorial_4.ipynb` or `tutorial_4.sh` exists

## Pass 1: Document CostGrow in the User Guide

Objective:

- replace the placeholder CostGrow section with a real user-facing explanation
- explain when to use CostGrow vs. ResUNet
- describe the expected inputs, runtime characteristics, and dependency requirements at a level appropriate for end users

Planned edits:

- expand the `CostGrow` section in `docs/user/user_guide.rst`
- add concise wording covering:
  - that CostGrow is rules-based, not learned weights
  - that it uses terrain-aware flood propagation on the high-resolution DEM
  - when it is a better fit than ResUNet
  - that PCRaster is required
  - that output interpretation and limits still depend on DEM quality and preprocessing

Manual test plan:

1. Open the updated section:
   - `sed -n '60,140p' docs/user/user_guide.rst`
2. Verify the placeholder text is gone.
3. Check the content for these concrete claims:
   - identifies `CostGrow_Terrain` by name
   - explains that it is rules-based
   - mentions PCRaster as a requirement
   - contrasts it with `ResUNet_16x_DEM`
4. Build or preview docs if desired:
   - `make -C docs/user html`
   - if that is not the standard local flow, use the existing project docs build command instead
5. Open the rendered user guide page and verify the CostGrow subsection appears in the Models section with clean formatting.

Expected verification results:

- the user guide presents CostGrow as a supported model rather than a placeholder
- no broken section formatting or malformed headings appear
- the page gives users enough context to choose between CostGrow and ResUNet

## Pass 2: Tighten Installation and CLI-Facing Docs

Objective:

- make sure a user who wants to run CostGrow can discover the prerequisites and relevant CLI path quickly
- keep the user docs aligned with the actual `tohr` interface

Planned edits:

- refine `docs/user/installation.rst` if the existing PCRaster guidance is too minimal
- update `docs/user/cli_reference.rst` text if needed so model selection for CostGrow is discoverable
- ensure any user-facing examples refer to the current `floodsr tohr` contract

Manual test plan:

1. Inspect the installation guidance:
   - `rg -n -C 2 "CostGrow|PCRaster" docs/user/installation.rst`
2. Inspect the CLI reference for `tohr` and model-related options:
   - `rg -n -C 2 "tohr|model-version|model-path|CostGrow" docs/user/cli_reference.rst`
3. Compare the docs against the actual parser help:
   - `python -m floodsr.cli tohr --help`
4. Verify the docs do not promise flags or defaults that are absent from the CLI output.
5. If docs were rebuilt, open the rendered installation and CLI reference pages and check:
   - CostGrow prerequisites are easy to find
   - the `tohr` examples render correctly

Expected verification results:

- a user can tell how to install dependencies needed for CostGrow
- a user can discover how to run `tohr` for a CostGrow workflow without reading source
- the docs remain consistent with `python -m floodsr.cli tohr --help`

## Pass 3: Add Tutorial 4 Skeleton

Objective:

- create the new tutorial notebook and hook it into the tutorials index
- establish the same execution/staging pattern used by tutorial 2

Planned edits:

- add `docs/user/notebooks/tutorial_4.ipynb`
- add `docs/user/notebooks/tutorial_4.sh`
- update `docs/user/tutorials.rst` to include tutorial 4 in the toctree
- use tutorial 2 as the structural template for:
  - notebook setup
  - imports
  - raster plotting helper(s)
  - shell execution cells
  - staged execution wrapper behavior

Manual test plan:

1. Confirm the new files exist:
   - `ls docs/user/notebooks/tutorial_4.ipynb docs/user/notebooks/tutorial_4.sh`
2. Confirm the tutorial index includes the new page:
   - `sed -n '1,220p' docs/user/tutorials.rst`
3. Open the notebook metadata quickly:
   - `python3 - <<'PY'
import json
from pathlib import Path
nb = json.loads(Path('docs/user/notebooks/tutorial_4.ipynb').read_text())
print(nb['cells'][0]['cell_type'])
print('cells=', len(nb['cells']))
PY`
4. Open the runner script:
   - `sed -n '1,240p' docs/user/notebooks/tutorial_4.sh`
5. Verify the script follows the same staging model as tutorial 2:
   - temp stage dir
   - local `floodsr` wrapper
   - `jupyter nbconvert --execute --inplace`

Expected verification results:

- tutorial 4 appears in the tutorials index
- the notebook is valid JSON and has a sensible top-level structure
- the runner script is aligned with the existing notebook execution conventions

## Pass 4: Build the CostGrow Tutorial Flow

Objective:

- turn tutorial 4 into a standalone CostGrow walkthrough
- keep the workflow quick enough for manual local validation

Planned edits:

- create tutorial content that:
  - downloads or references the same sample data pattern used in tutorial 2
  - introduces the purpose of CostGrow
  - runs `tohr` with the CostGrow model
  - explains the expected output files
  - adds simple plotting of inputs and the CostGrow result

Recommended notebook flow:

1. install/import notebook dependencies
2. download or resolve tutorial inputs
3. verify environment and CLI availability
4. run CostGrow `tohr`
5. validate output file creation
6. plot low-res input, DEM, and CostGrow output

Manual test plan:

1. Open the notebook and verify the markdown flow reads coherently from top to bottom.
2. Run the notebook execution wrapper:
   - `conda run -n dev bash docs/user/notebooks/tutorial_4.sh`
3. If `conda run -n dev` is not the right local environment, run the same script from the environment that already executes tutorial 2 successfully.
4. Confirm the executed notebook is written back into:
   - `docs/user/notebooks/tutorial_4.ipynb`
5. Confirm the expected output raster(s) are produced during execution by inspecting notebook cells or staged logs.
6. Review the final plots and verify:
   - the input low-res flood raster renders
   - the DEM renders
   - the CostGrow output renders without obvious plotting errors

Expected verification results:

- tutorial 4 runs end to end in the intended environment
- the notebook produces a CostGrow output raster
- the figures are understandable and visually consistent with tutorial 2 quality

## Pass 5: Add ResUNet Comparison Path

Objective:

- extend tutorial 4 so users can compare CostGrow output against a ResUNet baseline
- keep the comparison path intentionally lightweight and easy to reason about

Planned edits:

- add a second `tohr` run in tutorial 4 using the ResUNet backend
- fetch ResUNet weights as needed inside the notebook
- write the comparison output to a clearly named raster
- explain what the comparison is meant to show and what it is not meant to prove

Suggested output naming:

- `*_costgrow_sr.tif`
- `*_resunet_sr.tif`

Manual test plan:

1. Execute the notebook again:
   - `conda run -n dev bash docs/user/notebooks/tutorial_4.sh`
2. Verify the notebook fetches or resolves ResUNet weights successfully.
3. Confirm both output rasters are created.
4. From the repo root or notebook stage dir, verify file existence if needed:
   - `find . -name '*costgrow*sr.tif' -o -name '*resunet*sr.tif'`
5. Inspect the notebook cells and verify the comparison run clearly uses:
   - one CostGrow invocation
   - one ResUNet invocation
6. Verify the markdown explains why both outputs exist and how the reader should compare them.

Expected verification results:

- both models run from the same tutorial inputs
- output file names are unambiguous
- the tutorial communicates comparison intent clearly without overselling results

## Pass 6: Add Side-by-Side Comparison Plotting

Objective:

- visualize the CostGrow and ResUNet outputs together using the tutorial 2 plotting pattern
- optionally include a difference/comparison raster if it helps the explanation

Planned edits:

- reuse tutorial 2 plotting helpers and layout style
- add a side-by-side figure covering:
  - input low-res depth
  - CostGrow result
  - ResUNet result
- optionally add:
  - a difference raster (`CostGrow - ResUNet`)
  - wet-area percentages or other lightweight summary labels

Manual test plan:

1. Re-run tutorial 4:
   - `conda run -n dev bash docs/user/notebooks/tutorial_4.sh`
2. Open the executed notebook and inspect the final comparison figure.
3. Verify each subplot title is explicit and uses the right model names.
4. Check that color maps are sensible and not misleading:
   - flood depth views should use a depth-friendly map such as `Blues`
   - DEM views should use a terrain-style map when shown
5. If a difference raster is included, verify the markdown explains the sign convention.
6. Confirm there are no missing-file errors or notebook cells relying on hidden manual setup.

Expected verification results:

- the notebook ends with a clear, side-by-side comparison
- a reader can visually compare CostGrow and ResUNet without extra tooling
- any optional difference visualization is labeled clearly enough to avoid confusion

## Pass 7: Final Docs Integration and Smoke Test

Objective:

- make sure the new tutorial is discoverable and the docs set reads consistently as a whole
- finish with a lightweight end-to-end manual QA pass

Planned edits:

- tighten wording across touched pages after the notebook content stabilizes
- update any internal cross-references to tutorial 4
- remove stale placeholder language if any remains

Manual test plan:

1. Verify the branch diff only contains intended docs/notebook changes:
   - `git diff --stat origin/dev_cg...HEAD`
2. Rebuild the user docs:
   - `make -C docs/user html`
3. Open the rendered tutorials index and confirm tutorial 4 is listed.
4. Open the rendered user guide and confirm CostGrow text is no longer a placeholder.
5. Run the notebook one final time:
   - `conda run -n dev bash docs/user/notebooks/tutorial_4.sh`
6. Run a quick CLI sanity check:
   - `python -m floodsr.cli --version`
   - `python -m floodsr.cli tohr --help`
7. Confirm the final notebook content matches the rendered docs links and narrative.

Expected verification results:

- docs build without regressions introduced by this work
- tutorial 4 is visible from the main tutorials page
- the final narrative across installation, user guide, CLI reference, and tutorial 4 is consistent

## Suggested Commit Boundaries

If you want this branch to stay easy to review, use roughly these commit slices:

1. replace follow-up stub with implementation plan
2. user guide and installation/CLI docs updates
3. tutorial 4 notebook and runner skeleton
4. CostGrow notebook flow
5. ResUNet comparison cells and final side-by-side plotting
6. docs integration and final polish

## Risks To Watch

- CostGrow runtime may require environment setup not available in the default notebook environment
- tutorial runtime may become too slow if the comparison path uses large inputs
- output naming may become confusing if staged notebook paths and checked-in notebook paths diverge
- docs text may drift from the actual CLI if examples are written before final command validation
- plotting can become misleading if model outputs are compared with inconsistent masking or color scaling

## Exit Criteria

This follow-up is complete when all of the following are true:

- the user guide contains a real CostGrow section
- installation and CLI docs let a user discover how to run CostGrow
- tutorial 4 exists and is linked from the tutorials index
- tutorial 4 runs CostGrow successfully on sample data
- tutorial 4 runs a ResUNet comparison path on the same sample data
- tutorial 4 ends with a side-by-side comparison figure
- the manual test steps above have been executed and verified
