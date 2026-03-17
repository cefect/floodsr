# syntax=docker/dockerfile:1.7

FROM condaforge/miniforge3:25.3.1-0

# Keep the image minimal: only install the notebook runtime and CLI helpers.
ENV PIPX_HOME=/opt/pipx
ENV PIPX_BIN_DIR=/opt/pipx/bin
ENV PATH=/opt/pipx/bin:${PATH}

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends curl

RUN python -m pip install --no-cache-dir pipx \
    && python -m pipx install --force --include-deps jupyter

WORKDIR /workspace

ENTRYPOINT ["/bin/bash", "-lc"]
