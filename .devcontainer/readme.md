# Dev containers

This repo has two devcontainer profiles under `.devcontainer/`.

## Main profile

- `.devcontainer/main/devcontainer.json`
- `.devcontainer/main/docker-compose.yml`
- `container/Dockerfile` (target: `dev`)

## Docs profile

- `.devcontainer/docs/devcontainer.json`
- `.devcontainer/docs/docker-compose.yml`
- `docs/container/Dockerfile` (target: `dev`)

## Manual image builds

```bash
docker buildx build --load -t cefect/floodsr:dev-v0.2 -f container/Dockerfile --target dev .
docker buildx build --load -t cefect/floodsr-docs:dev-v0.1 -f docs/container/Dockerfile --target dev .
```
