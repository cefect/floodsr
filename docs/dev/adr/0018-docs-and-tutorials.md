# ADR-0018: Docs and Tutorials Strategy

## Context

The user-facing docs now include tutorial content alongside the CLI reference and narrative guides. 

The current docs stack uses Sphinx with `myst_nb` for notebook rendering and the Read the Docs theme for HTML output.

## Decision

- Keep user docs under `docs/user`.
- Add a dedicated tutorials landing page at `docs/user/tutorials.rst`.
- Author tutorial notebooks as real `.ipynb` files under `docs/user/notebooks/`.
- Render tutorial notebooks into the docs site with `myst_nb`.
- Keep notebook execution disabled during docs builds.
- Treat tutorial notebooks as documentation artifacts first, not as build-time executed tests.
- Track any future interactive-launch button as a separate implementation issue rather than in this ADR.
- Keep English as the source language for user docs.
- Add Canadian French (`fr_CA`) as a translated docs target using Sphinx i18n catalogs under `docs/user/locale/`.
- Keep one shared Sphinx source tree under `docs/user` rather than duplicating content per language.
- Keep the docs in the main repo and publish translations from separate Read the Docs projects that point at the same repository.

## Localization

- The canonical authored docs remain the English files already stored under `docs/user`.
- French content should be maintained as `.po` catalogs under `docs/user/locale/fr_CA/LC_MESSAGES/`.
- Compiled `.mo` files are build artifacts, not source artifacts. They should be generated for local/CI/docs builds and should not be committed to the repository.
- Translate top-level indexes, landing pages, and internal links along with page content so the French docs are navigable as a complete experience.

### translator instructions
- target Canadian French (`fr_CA`)
- Do not use `gettext` to generate translation files. Create the `.po` files directly, with a best-effort translation that preserves the tone and meaning of the English source.
- After editing `.po` files, compile them to `.mo` files for the build/review step, but do not treat the compiled `.mo` files as tracked source files.
- Keep commands, code, stdout, and project names unchanged, including `HRDEM` and `CostGrow`.
- When a term should stay tied to the English wording, explain it in French rather than forcing a literal translation. For example, `to high resolution (tohr)` should note the English phrase in the French text.
- Critically review translations for readability and fidelity rather than translating mechanically.
- Build the docs after translation work and review the rendered result to confirm the translation, navigation, and links behave as intended.

## Tutorials

 
- `docs/user/tutorials.rst` is the landing page for curated walkthroughs.
- Each tutorial should have:
  - a short title
  - a one-line description on the landing page
- Tutorial numbering should be explicit and stable, e.g. `Tutorial 1: Quick Start`.
- each tutorial notebook should be runnable in the *dev* layer of the .devcontainer (`container/miniforge/environment.dev.yml`). this includes setup tutorials (see below). Tutorial execution for documentation refreshes should use per-notebook shell shims that live beside the notebooks under `docs/user/notebooks/` (for example `tutorial_1.sh` and `tutorial_2.sh`).

### setup tutorials (1 and 2)
As these provide commands for patching the environment, they are a special case. 

- the same notebook should support the user in all 3 execution contexts through .md instructions, cross-links tot he rst docs, and commented-out cells for notebook users.
- runnability: per-above, these notebooks ALSO need to run inside the dev environment. we accomplish this by commenting out install commands in the notebook cells and providing instructions to un-comment if needed.
- Each setup tutorial notebook (i.e., 1 and 2) should start with a short summary of the three execution contexts:
  - command line (CLI)
  - local notebook (Jupyter)
  - hosted notebook (Colab)
- That summary should link readers to the corresponding section of the installation page rather than duplicating the full install guidance inline.
- Each tutorial notebook should adopt the user flow described for install/setup in the docs, which depends on the execution environment.
  - notebook, this is a mix of CLI and notebook cells with guidance for both local and hosted notebook users.
  - notebook cell install commands should be commented out, with instructions to un-comment if applicable.
  - CLI, this is a pure CLI flow with instructions to switch to the notebook if desired.

### tutorial execution assumptions

- Tutorials should invoke `floodsr` through the CLI, including from notebooks (for example `!floodsr ...`), rather than switching to the Python API unless the tutorial explicitly says otherwise.
- User-facing notebook tutorials should prefer literal notebook CLI cells (for example `!floodsr ...`) over Python `subprocess` wrappers so the commands remain easy to read, copy, and adapt.
- Tutorials >1 require some **additional dependencies** (`matplotlib` and `rasterio`) in addition to `floodsr`.
- Tutorials >1 should print the versions of additional dependencies near the start of the notebook so users can quickly confirm their runtime.
  - For the CLI path in Tutorials >1, the docs should distinguish between the `floodsr` `pipx` install and the Python environment used for notebook execution and plotting.
  - CLI guidance for Tutorials >1 should remain intentionally flexible: if the user is following the CLI-only path, they should use an existing Python environment, or create a new one with conda or venv, then installdependencies there.
  - The CLI path should remind users to copy and paste commands into their terminal, and note that plotting commands may need small revisions to save figures to disk instead of displaying them inline, or may simply be skipped.
  - For hosted notebook (Colab), Tutorials >1 may assume additional dependencies are already available 
  - For local notebook (Jupyter), Tutorials >1 should include a commented-out `%pip install matplotlib rasterio` cell so users can patch the active kernel environment by uncommenting it when needed.

 for testing, see https://github.com/cefect/floodsr/issues/31

## Tutorial Notebook Rendering Strategy

- Tutorial source files should be committed as real Jupyter notebooks (`.ipynb`), not generated artifacts and not Markdown stand-ins.
- Notebook files should live under `docs/user/notebooks/`.
- Sphinx should render these notebooks with `myst_nb`.
- Notebook execution should remain disabled via the docs configuration so that builds are deterministic
- Tutorial execution for documentation refreshes should use per-notebook shell shims that live beside the notebooks under `docs/user/notebooks/` (for example `tutorial_1.sh` and `tutorial_2.sh`).
- Those shell shims should execute a temporary copy of the notebook under a cache-backed working directory, then copy the completed `.ipynb` back into `docs/user/notebooks/`.
- Generated side files from tutorial execution should stay in the cache-backed staging area, not beside the tracked source notebooks.
- Per-notebook shell shims should assume the caller has already activated the correct notebook runtime. In this repo, proofing should be launched from the outside with `conda run -n dev ...` (or an already-active `dev` shell) rather than hard-coding a conda interpreter path inside the shim.
- Notebook source cells should default to the same cache behavior as the application code. For `floodsr`, that means leaving cache selection to the CLI/runtime unless the user explicitly edits the notebook cell to override it.
- When a tutorial benefits from cache reuse during docs proofing (for example, repeated HRDEM fetches in Tutorial 3), the per-notebook shell shim may inject a local shared-cache override via environment variables. That override should live in the shim, not as a hard-coded path in the committed notebook source.
- Tutorials that expose cache overrides should tell users to edit the relevant notebook cell if they want custom cache behavior.
 

## Docs Devcontainer Limitation

Tutorial notebooks are rendered by the docs toolchain, but they are not expected to run inside the docs devcontainer as part of normal docs authoring.

Notebook proofing may still be added under `pytest` as a separate `notebook`-marked suite, but that execution should remain outside the default `deploy` test path and should run from the `dev` conda environment so the notebook runtime stays isolated from the runtime-locked package environment.
