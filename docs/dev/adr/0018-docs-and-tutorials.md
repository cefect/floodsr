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
- French content should be maintained as `.po` translation catalogs under `docs/user/locale/fr_CA/LC_MESSAGES/`.
 

 

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

 

## Tutorial Notebook Rendering Strategy

- Tutorial source files should be committed as real Jupyter notebooks (`.ipynb`), not generated artifacts and not Markdown stand-ins.
- Notebook files should live under `docs/user/notebooks/`.
- Sphinx should render these notebooks with `myst_nb`.
- Notebook execution should remain disabled via the docs configuration so that builds are deterministic
- Tutorial execution for documentation refreshes should use per-notebook shell shims that live beside the notebooks under `docs/user/notebooks/` (for example `tutorial_1.sh` and `tutorial_2.sh`).
- Those shell shims should execute a temporary copy of the notebook under a cache-backed working directory, then copy the completed `.ipynb` back into `docs/user/notebooks/`.
- Generated side files from tutorial execution should stay in the cache-backed staging area, not beside the tracked source notebooks.
 

## Docs Devcontainer Limitation

Tutorial notebooks are rendered by the docs toolchain, but they are not expected to run inside the docs devcontainer as part of normal docs authoring.

Notebook proofing may still be added under `pytest` as a separate `notebook`-marked suite, but that execution should remain outside the default `deploy` test path and should run from the `dev` conda environment so the notebook runtime stays isolated from the runtime-locked package environment.

