# Contributing

[project plan](https://docs.google.com/document/d/1_QnUurhdNyuawVcDFsKNLw6biA8Yw130GBm-3e6ij9o/edit?usp=sharing)

## patch dev environment
```bash
# patch floodsr cli
floodsr() { python -m floodsr.cli "$@"; }
export -f floodsr

# check the version of floodsr
floodsr --version
```


## .devcontainer setup
use `.devcontainer/main` for code development. this profile needs to:
- have the `FLOODSR_GITHUB_TOKEN` environment variable set or the `gh` CLI authenticated to access private repo assets (see "Auth model" below).


`.devcontainer/main/devcontainer.json` example:
```json
{
  "dockerComposeFile": "./docker-compose.yml",
  "service": "dev",
  "workspaceFolder": "/workspace",
  "shutdownAction": "stopCompose",
  "updateRemoteUserUID": true,
  "containerUser": "cefect",
  "remoteUser": "cefect",
  "containerEnv": {
    "CODEX_HOME": "/home/cefect/.codex",
    "FLOODSR_GITHUB_TOKEN": "${localEnv:FLOODSR_GITHUB_TOKEN}",
    "GITHUB_TOKEN": "${localEnv:GITHUB_TOKEN}",
    "GH_TOKEN": "${localEnv:GH_TOKEN}"
  },
  "customizations": {
    "vscode": {
      "settings": {
        "python.defaultInterpreterPath": "/opt/conda/envs/deploy/bin/python",
        "python.terminal.activateEnvironment": false, //already set on the container
        "python.useEnvironmentsExtension": false
      }
    }
  }
}
```

`.devcontainer/main/docker-compose.yml` example:
```yaml
name: floodsr_compose
services:
  dev:
    image: cefect/floodsr:miniforge-dev-v0.9
    environment:
      TMPDIR: /home/cefect/LS/10_IO/2407_FHIMP/tmp
      XDG_CONFIG_HOME: /home/cefect/.config
      PYTHONPATH: /workspace
    volumes:
      - /home/cefect/LS/09_REPOS/04_TOOLS/floodsr:/workspace:delegated
      - /home/cefect/LS/10_IO/2407_FHIMP:/home/cefect/LS/10_IO/2407_FHIMP:delegated
      - /home/cefect/.config:/home/cefect/.config:rw
      - /home/cefect/.ssh:/home/cefect/.ssh:ro
      - /home/cefect/.codex:/home/cefect/.codex:rw
      - /home/cefect/.pypirc:/home/cefect/.pypirc:ro
      - /etc/localtime:/etc/localtime:ro
      - /etc/timezone:/etc/timezone:ro
    working_dir: /workspace
    user: 1000:1000
    tty: true
    stdin_open: true
    cpus: 8
    mem_limit: 24g
    pid: host
    command:
      - sleep infinity

```

The repo also includes a separate docs profile under `.devcontainer/docs`, but all code and tests should run from the `main` profile with the `deploy` conda environment.


## Development Environment Setup

 

The `dev-setup.sh` script sets up the development environment using Docker:

```bash
chmod +x dev-setup.sh && ./dev-setup.sh
```

### What it does

1. Checks prerequisites (docker, git, git-lfs, gh)
2. Authenticates with GitHub and exports `FLOODSR_GITHUB_TOKEN`
3. Fetches Git LFS test data
4. **Builds a Docker image for x86_64 (`linux/amd64`) regardless of host architecture**
5. Validates that key packages (numpy, rasterio, pydantic) can be imported inside the container

### Platform support

The Docker image is always built for x86_64 (`--platform linux/amd64`):
- **On x86_64 hosts**: Builds natively, then validates package imports
- **On ARM64 hosts** (Apple Silicon, M1/M2/M3 Macs): Uses Docker's QEMU emulation for cross-compilation, then validates package imports

This ensures consistency across platforms since pcraster only has x86_64 builds on conda-forge. The validation test runs on all platforms inside the x86_64 container to verify key packages are importable.

### After setup completes

Run an interactive dev shell:

```bash
docker run --rm -it \
  --entrypoint /bin/bash \
  -v "$PWD:/workspace" \
  -e FLOODSR_GITHUB_TOKEN="$(gh auth token)" \
  -w /workspace \
  cefect/floodsr:miniforge-dev-v0.9 -l
```

Then verify inside the container:

```bash
python -m floodsr.cli models list
python -m floodsr.cli models fetch ResUNet_16x_DEM
pytest -q tests/test_model_registry.py
```



## Auth model used by this project

- Git repository operations use SSH (`git@github.com:cefect/floodsr.git`).
- Model artifact fetches use HTTPS release URLs from `floodsr/models.json`.
- While the repo/releases are private, HTTPS fetches require a GitHub token.
- After public release assets are enabled, token auth should be optional.

### One-time setup (private phase)

1. Authenticate the GitHub CLI on the host (`gh auth login`).
2. export the GitHub token to an environment variable (`FLOODSR_GITHUB_TOKEN`).
 
```bash
export FLOODSR_GITHUB_TOKEN="$(gh auth token)"
 
```

If you use `.devcontainer`, wire this variable through container env (for example with
`containerEnv`/`remoteEnv` in `.devcontainer/devcontainer.json`, or via
`.devcontainer/docker-compose.yml`) so `FLOODSR_GITHUB_TOKEN` is available inside the
container.
 
### Verify model fetch and link checks

```bash
python -m floodsr.cli models list
python -m floodsr.cli models fetch ResUNet_16x_DEM
pytest -q tests/test_model_registry.py::test_default_manifest_http_links_resolve
```

## Fetch test TIFFs from Git LFS

`tests/data/*.tif` is tracked by Git LFS. If you cloned with LFS smudge disabled, you may have pointer text files instead of GeoTIFF binaries.

 
