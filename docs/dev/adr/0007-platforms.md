# ADR-0007: Platform and Execution Context Support

CPU-first is officially supported. GPU may be considered in the future. 

## Context

`floodsr` now has two capability tiers defined in `ADR-0002`:
- `basic`
- `extended`

Those tiers are not available in every environment. The project also runs in several distinct execution contexts with different install and support constraints:
- command line (CLI)
- local notebook (Jupyter)
- hosted notebook (Colab)

 

## Decision

We track support across three dimensions:
- Capability tier: `basic`, `extended`
- Platform: `Linux`, `Windows`, `macOS`
- Execution context: `command line (CLI)`, `local notebook (Jupyter)`, `hosted notebook (Colab)`

## Support Matrix
 
- command line (CLI)
  - `basic`: supported
  - `extended`: supported
- local notebook (Jupyter)
  - `basic`: supported
  - `extended`: supported
- hosted notebook (Colab)
  - `basic`: supported
  - `extended`: **experimental**. see note below

 

### macOS

not implemented yet

## Notes

- `basic` is the default capability tier for the broadest user base.
- `basic` assumes the user already has a working Python installation in the target execution context.
- `extended` requires a conda-managed environment with GDAL already installed before `pip install floodsr`.
- `command line (CLI)` means a local shell-driven workflow and should recommend `pipx` for the `basic` install path.
- `local notebook (Jupyter)` means a notebook kernel launched from a user-managed local environment and assumes the user already has Jupyter working there.
- `local notebook (Jupyter)` should recommend `pip` into the environment that backs the selected kernel, not `pipx`.
- `hosted notebook (Colab)` means installs occur inside a managed Colab runtime and should use `pip` inside the notebook session.

### colab + gdal
there are some hacky ways to get gdal to work with colab, that will probably break at some point when the colab backends are updated.
This setup seems to work: https://colab.research.google.com/drive/1QCdoW1_MZU_eWucLaUMjxf5osOxr94v0?usp=sharing

let's include these for now with a warning. 





## Consequences

- User docs should describe capability tier separately from platform and execution context.
- Command line (CLI) and local Jupyter guidance may share the same underlying Python environment when the capability tier is the same, but their recommended install commands differ.
- Colab guidance should stay limited to the `basic` tier unless hosted `extended` support becomes an explicit project goal.
- If platform support changes, update this ADR rather than broadening packaging claims in `ADR-0002`.

## Related Decisions

- See `ADR-0002` for the packaging and install strategy behind `basic` and `extended`.
