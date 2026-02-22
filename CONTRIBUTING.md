# Contributing

## .devcontainer setup
needs to:
- have the `FLOODSR_GITHUB_TOKEN` environment variable set or the `gh` CLI authenticated to access private repo assets (see "Auth model" below).


.devcontainer/devcontainer.json example:
```json
{
   
  "dockerComposeFile": "./docker-compose.yml",
  
  "service": "dev",
  "workspaceFolder": "/workspace",
  "shutdownAction": "stopCompose",
  
  "containerEnv": {
 
    "FLOODSR_GITHUB_TOKEN": "${localEnv:FLOODSR_GITHUB_TOKEN}",
    "GITHUB_TOKEN": "${localEnv:GITHUB_TOKEN}",
    "GH_TOKEN": "${localEnv:GH_TOKEN}"

  },
   
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python",
        "python.terminal.activateEnvironment": false, //already set on the container
        "python.testing.pytestEnabled": true,
 
      }
    }
  }
}
```

.devcontainer/docker-compose.yml example:
```yaml
name: floodsr_compose
services:
  dev:
    build:
      context: ../..
      dockerfile: container/Dockerfile
      target: dev
    image: cefect/floodsr:dev-v0.3
    environment:
      TMPDIR: /home/cefect/LS/10_IO/2407_FHIMP/tmp
      XDG_CONFIG_HOME: /home/cefect/.config
      PYTHONPATH: /workspace
    volumes:
      - ../..:/workspace:delegated
      - /home/cefect/LS/10_IO/2407_FHIMP:/home/cefect/LS/10_IO/2407_FHIMP:delegated
      - /home/cefect/LS/10_IO/2407_FHIMP/tmp:/home/cefect/LS/10_IO/2407_FHIMP/tmp:rw
      # Mount host user config so `gh auth` state is visible inside container.
      - ${HOME}/.config:/home/cefect/.config:rw
      # Mount host SSH keys/agent sockets for git@github.com workflows.
      - ${HOME}/.ssh:/home/cefect/.ssh:rw
      # Mount codex home from the active host user.
      - ${HOME}/.codex:/home/cefect/.codex:rw
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

# PIPX local install

```bash
pipx uninstall floodsr || true
pipx install --force .

floodsr --help
floodsr doctor
floodsr models list

floodsr infer \
  --in tests/data/2407_FHIMP_tile/lowres032.tif \
  --dem tests/data/2407_FHIMP_tile/hires002_dem.tif

```