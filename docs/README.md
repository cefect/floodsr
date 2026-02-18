# Docs workspace

This folder contains the Sphinx + Read the Docs setup for `floodsr`.

## Files

- `docs/conf.py`: Sphinx configuration.
- `docs/index.rst`: current single-page starter content.
- `docs/_templates/layout.html`: template override for light branding.
- `docs/_static/custom.css`: theme-level style overrides.
- `docs/.devcontainer/`: docs-focused VS Code dev container.
- `docs/container/`: docs Dockerfile and requirements overlays.

## Edit and build from inside container

1. Open the `docs/` folder in VS Code.
2. Run **Dev Containers: Reopen in Container**.
3. Confirm current working directory is `/workspace/docs`.
4. Build docs:

```bash
sphinx-build -b html . _build/html
```

5. Optional live preview:

```bash
sphinx-autobuild . _build/html --host 0.0.0.0 --port 8000
```

## Read the Docs setup

This repo now includes `.readthedocs.yaml` at project root.

1. Push this branch to your remote.
2. In Read the Docs, use **Import a Project** for this repository.
3. Keep the default config path: `.readthedocs.yaml`.
4. Trigger a build and verify `docs/index.rst` renders.
