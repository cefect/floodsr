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



# PLAN


- getting started
    -  'what is', 
    - 'installation' (just the basic pipx command and a link to the install page), 
    -  'quickstart' (with a very simple example of running `floodsr tohr` on the test 'tests/data/2407_FHIMP_tile')
    - `FAQ`: blank for now
- installation
    - system requirements (hardware, OS, etc)
    - install with pipx (recommended)
    - install with pip (advanced)
    - install from source (advanced)
- User Guide
    - introduction: paragraph on resolution enhamcenet, terminology, what the tool does, how it might be used
    - CLI reference (auto-generated from `floodsr --help` and subcommands)
    - models: one sub-section for each model. 