# Docs workspace

This folder contains user documentation and developer architecture records.

## Structure

- `docs/user/`: user-facing docs + RTD/Sphinx implementation and docs container.
- `docs/dev/adr/`: architectural decision records (ADRs).

## Build user docs

1. Build the docs image (dev target):

```bash
docker buildx build --load -t floodsr-docs:dev -f container/docs/Dockerfile --target dev .
```

2. Build HTML:

```bash
sphinx-build -b html docs/user docs/user/_build/html
```

## Read the Docs config

- `docs/user/.readthedocs.yaml`
