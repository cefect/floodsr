# sandbox environments in devcontainer

we use two devcontainer environments:
- `main` for dev and source code (`.devcontainer/main/devcontainer.json` points to `container/miniforge/Dockerfile`)
- `docs` for sphinx documentation work (`.devcontainer/docs/devcontainer.json` points to `container/docs/Dockerfile`)

Also, some dev work may happen outside of the containers (e.g., in simple `wsl`). 

Before running any command, probe your sandbox to understand which environment you are in.

## main environment
- all code and tests should run here with the `deploy` conda environment. (see `container/miniforge/conda-env-deploy.lock.yml`). This mirrors the production environment and should be used for all development work, including running tests.
- the `dev` conda environment is layered on top of this and should only be used for running notebooks (see `container/miniforge/conda-env-dev.lock.yml`)


# user documentation


## maintenance

### translation
- only perform translation updates/maintenance when excplitly asked. 
- refresh the full fr docs translation per ADR-0018: update the existing French .po catalogs directly without gettext regeneration, compile the catalogs.
- fix any simple/obvious english typos/errors.
- only touch/review .po files whose english source has changed since the last translation refresh.
- rebuild the French HTML. review the rendered result for navigation/links/readability, and make any follow-up fixes needed to match policy. ensure everything intended translates. (no typography artifacts).