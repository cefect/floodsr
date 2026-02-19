# floodsr

Super-Resolution for flood hazard rasters.


## Installation

```bash
pip install -e ".[dev]"
```


## Use

The Phase-4 CLI surface is focused on model registry operations.

List available model versions:

```bash
# dev shortcut for the CLI
alias floodsr='python -m floodsr.cli'

floodsr models list
```

Fetch a model by version into the default cache:

```bash
floodsr models fetch 4690176_0_1770580046_train_base_16
```

Fetch using explicit options:

```bash
floodsr models fetch 4690176_0_1770580046_train_base_16 \
  --cache-dir ./tmp/model-cache \
  --backend file \
  --manifest ./floodsr/models.json
```

Notes:
- `models list` supports `--manifest <path>` to use a custom manifest.
- `models fetch` supports `--manifest`, `--cache-dir`, `--backend {http,file}`, and `--force`.
- for private GitHub release assets, set `FLOODSR_GITHUB_TOKEN` (or `GITHUB_TOKEN` / `GH_TOKEN`) before `models fetch`.
- `.devcontainer/main` and `.devcontainer/docs` pass these token env vars from host to container.
- If the `floodsr` console script is not installed yet, use:

```bash
python -m floodsr.cli models list
```



 
