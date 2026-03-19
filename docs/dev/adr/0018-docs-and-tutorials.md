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

## Tutorials

 
- `docs/user/tutorials.rst` is the landing page for curated walkthroughs.
- Each tutorial should have:
  - a short title
  - a one-line description on the landing page
- Tutorial numbering should be explicit and stable, e.g. `Tutorial 1: Quick Start`.
- each tutorial notebook should be runnable in the *dev* layer of the .devcontainer (`container/miniforge/environment.dev.yml`). this includes setup tutorials (see below). 

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

 

## Notebook Rendering Strategy

- Tutorial source files should be committed as real Jupyter notebooks (`.ipynb`), not generated artifacts and not Markdown stand-ins.
- Notebook files should live under `docs/user/notebooks/`.
- Sphinx should render these notebooks with `myst_nb`.
- Notebook execution should remain disabled via the docs configuration so that builds are deterministic and do not depend on runtime services, package installation side effects, network availability, or notebook kernel behavior.
 

## Docs Devcontainer Limitation

Tutorial notebooks are rendered by the docs toolchain, but they are not expected to run inside the docs devcontainer as part of normal docs authoring.

Notebook proofing may still be added under `pytest` as a separate `notebook`-marked suite, but that execution should remain outside the default `deploy` test path and should run from the `dev` conda environment so the notebook runtime stays isolated from the runtime-locked package environment.

Future work for a Colab-backed interactive launch button is tracked in issue `#18`.

## Consequences

### Positive

- Tutorials stay close to the user docs and are easy to discover.
- Notebook tutorials are authored in a format users can download and reuse directly.
- Docs builds remain stable because notebooks are rendered, not executed.

### Negative

- Notebook rendering and notebook execution are intentionally separated, which may confuse contributors at first.
- The docs devcontainer is not a guaranteed runtime for tutorial execution.

## Implementation Notes

- Keep tutorial notebooks small and task-oriented.
- Prefer CLI-first tutorial flows that match the existing user docs.
- Keep install/setup cells aligned with the execution-context split in `ADR-0007`: `pipx` for CLI, `pip` for notebook kernels, and separate hosted-notebook guidance.
- Avoid introducing notebook outputs into version control unless there is a strong reason to preserve them.
- If interactive execution becomes a core product feature later, revisit whether Colab remains sufficient or whether a more integrated notebook hosting model is needed.
