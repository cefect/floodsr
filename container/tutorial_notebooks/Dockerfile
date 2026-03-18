# syntax=docker/dockerfile:1.7

FROM condaforge/miniforge3:25.3.1-0

# Keep the image minimal: only install the notebook runtime and CLI helpers.

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends curl

RUN python -m pip install --no-cache-dir jupyter

WORKDIR /workspace

ENTRYPOINT ["/bin/bash", "-lc"]
