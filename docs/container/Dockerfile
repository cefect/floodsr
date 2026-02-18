# syntax=docker/dockerfile:1.7

ARG USERNAME=cefect
ARG USER_UID=1000
ARG USER_GID=1000

FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm AS base

# Avoid interactive prompts.
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system packages needed for Sphinx docs workflows.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    make \
    && rm -rf /var/lib/apt/lists/*

# Capture base image Python requirements as constraints for minimal-diff overlays.
RUN python -m pip freeze --all > /tmp/requirements.base.txt

# Copy overlay-only dependencies.
COPY docs/container/requirements.overlay.txt /tmp/requirements.overlay.txt

# Install docs packages while constraining base versions.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
    --constraint /tmp/requirements.base.txt \
    --requirement /tmp/requirements.overlay.txt

# Capture final environment for inspection/debugging in the built image.
RUN python -m pip freeze --all > /tmp/requirements.final.txt

########################
# Dev
########################

FROM base AS dev

# declare args again for this stage
ARG USERNAME
ARG USER_UID
ARG USER_GID
ARG OLD_USERNAME=vscode

# Docs authoring layer: keep base package versions pinned via freeze constraints.
COPY docs/container/requirements.overlay.dev.txt /tmp/requirements.overlay.dev.txt
RUN python -m pip freeze --all > /tmp/requirements.base-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
    --constraint /tmp/requirements.base-dev.txt \
    --requirement /tmp/requirements.overlay.dev.txt
RUN python -m pip freeze --all > /tmp/requirements.dev.txt

# Rename existing user if present; otherwise create the dev user.
RUN set -eux; \
    if id -u "${OLD_USERNAME}" >/dev/null 2>&1 && [ "${OLD_USERNAME}" != "${USERNAME}" ]; then \
        usermod -l "${USERNAME}" "${OLD_USERNAME}"; \
        usermod -d "/home/${USERNAME}" -m "${USERNAME}"; \
        groupmod -n "${USERNAME}" "${OLD_USERNAME}" || true; \
    elif ! id -u "${USERNAME}" >/dev/null 2>&1; then \
        groupadd --gid "${USER_GID}" "${USERNAME}"; \
        useradd  --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USERNAME}"; \
    fi; \
    usermod -u "${USER_UID}" "${USERNAME}" || true; \
    groupmod -g "${USER_GID}" "${USERNAME}" || true

# Add sudo + common CLI tooling.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        sudo \
        git \
        git-lfs \
        openssh-client \
        wget \
        ca-certificates \
        tar \
        curl \
        rsync \
        procps \
        strace \
    && rm -rf /var/lib/apt/lists/*

RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-${USERNAME} \
    && chmod 0440 /etc/sudoers.d/90-${USERNAME}

# Give the dev user ownership of places they'll modify.
RUN set -eux; \
    install -d -o ${USER_UID} -g ${USER_GID} -m 0755 /workspace; \
    install -d -o ${USER_UID} -g ${USER_GID} -m 1777 /tmp /var/tmp; \
    mkdir -p /home/${USERNAME}; \
    chown -R ${USER_UID}:${USER_GID} /home/${USERNAME}

# Share the same global git configuration as the main dev image.
COPY container/etc-gitconfig /etc/gitconfig
RUN chmod 644 /etc/gitconfig

# set default user
USER ${USER_UID}:${USER_GID}
WORKDIR /workspace

ENTRYPOINT ["/bin/bash", "-lc"]
