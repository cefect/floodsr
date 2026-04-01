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
- Add French (`fr`) as a translated docs target using Sphinx i18n catalogs under `docs/user/locale/`.
- Keep one shared Sphinx source tree under `docs/user` rather than duplicating content per language.
- Keep the docs in the main repo and publish translations from separate Read the Docs projects that point at the same repository.
- Treat the landing pages, getting-started pages, installation pages, and tutorial introductions as beginner-facing content for readers with minimal programming and GIS knowledge.
- Treat the Python CLI parser definition as the source of truth for CLI reference documentation, while maintaining the committed docs page as a manually refreshed artifact rather than regenerating it during docs builds.

## French docs

- The canonical authored docs remain the English files  `docs/user`.
- French content should be maintained as `.po` catalogs under `docs/user/locale/fr/LC_MESSAGES/`.
- Compiled `.mo` files are build artifacts, not source artifacts. They should be generated for local/CI/docs builds and should not be committed to the repository.
- Translate top-level indexes, landing pages, and internal links along with page content so the French docs are navigable as a complete experience.

### translator instructions
- target French (`fr`)
- Do not use `gettext` to generate translation files. Create the `.po` files directly, with a best-effort translation that preserves the tone and meaning of the English source.
- After editing `.po` files, compile them to `.mo` files for the build/review step, but do not treat the compiled `.mo` files as tracked source files.
- Keep commands, code, stdout, and project names unchanged, including `HRDEM` and `CostGrow`.
- In `cli_reference.po`, translate the narrative help text and explanatory prose, but do not translate literal commands, subcommands, flags, option names, paths, or code-like tokens.
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

## CLI Reference Strategy

- `docs/user/cli_reference.rst` should be refreshed from parser metadata exposed by the Python CLI, not by capturing terminal help output into one literal code block.
- The CLI parser should remain the canonical definition of commands, arguments, defaults, choices, and help text.
- Generated CLI docs should keep commands, flags, metavars, and examples literal, while emitting command descriptions and option help as normal translatable prose so `cli_reference.po` can localize them.
- Docs builds should read the committed `docs/user/cli_reference.rst` artifact directly and should not try to regenerate it on Read the Docs or other build hosts.
- Refreshing `docs/user/cli_reference.rst` is therefore a manual maintenance step performed by a developer in a runtime where `floodsr` is importable.

## Tutorial Notebook Rendering Strategy

- Tutorial source files should be committed as real Jupyter notebooks (`.ipynb`), not generated artifacts and not Markdown stand-ins.
- Notebook files should live under `docs/user/notebooks/`.
- Sphinx should render these notebooks with `myst_nb`.
- Notebook execution should remain disabled via the docs configuration so that builds are deterministic.
- Tutorial execution for documentation refreshes should use per-notebook shell shims that live beside the notebooks under `docs/user/notebooks/` (for example `tutorial_1.sh` and `tutorial_2.sh`).
- Those shell shims should execute a temporary copy of the notebook under a temp-backed sandbox-like working directory, then copy the completed `.ipynb` back into `docs/user/notebooks/`.
- CI install-path proof should stay lightweight: `install-edge.yml` may execute small notebook shim artifacts that mirror the documented notebook install commands and a minimal `floodsr` sanity check, rather than the full tutorial notebooks.
- The full tutorial notebooks should continue to be proven separately via the `notebook`-marked pytest suite and the per-notebook shell runners used for docs refreshes.
- Generated side files from tutorial execution should stay in that temp-backed staging area, not beside the tracked source notebooks.
- The committed notebook artifacts under `docs/user/notebooks/` should be pruned before rendering so they keep plot/image outputs that materially help the docs, while dropping textual execution output such as stream logs, CLI chatter, and one-off diagnostics.
- The docs site should therefore render tutorial notebooks from this pruned state: markdown plus code cells, with plot outputs preserved where useful and non-plot outputs removed.
- Short notebook-internal validation cells may remain executable while being hidden from rendered docs by using notebook cell tags such as `remove-input` for assertion-only checks.
- When a notebook cell uses `remove-output`, add one short preceding code cell tagged `remove-input` with a plain editor-facing note such as `# cell below has tag:'remove-output'` so the hidden-output behavior is obvious while editing the notebook in VS Code, without changing what readers see in the rendered docs.
- Per-notebook shell shims should assume the caller has already activated the correct notebook runtime. In this repo, proofing should be launched from the outside with `conda run -n dev ...` (or an already-active `dev` shell) rather than hard-coding a conda interpreter path inside the shim.
- Notebook source cells may define a visible, hard-coded `base_cache_dir` when that keeps the tutorial easier to read and rerun.
- When a tutorial uses a visible `base_cache_dir`, add a hidden follow-up cell that lets docs-proofing or CI override that path from environment variables without changing the user-facing flow.
- When a tutorial benefits from cache reuse during docs proofing (for example, repeated HRDEM fetches in Tutorial 3), the per-notebook shell shim may still inject the cache path via environment variables, but the notebook should resolve that through the hidden override cell rather than through ad hoc command-string assembly later in the tutorial.
- Tutorials that expose cache overrides should tell users to edit the visible notebook cache cell if they want custom cache behavior.
- When docs are previewed from a non-`main` branch, the Colab launch button may therefore open an older `main` branch notebook rather than the previewed content.
 

## Docs Devcontainer Limitation

Tutorial notebooks are rendered by the docs toolchain, but they are not expected to run inside the docs devcontainer as part of normal docs authoring.

Notebook proofing may still be added under `pytest` as a separate `notebook`-marked suite, but that execution should remain outside the default `deploy` test path and should run from the `dev` conda environment so the notebook runtime stays isolated from the runtime-locked package environment.
