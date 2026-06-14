#!/usr/bin/env bash
set -euo pipefail

# Run all tutorial notebook refresh shims from WSL in a one-off container that
# mirrors the main devcontainer resources without using `docker compose`.
#
# Usage:
# - From WSL at the repo root:
#   `bash docs/user/notebooks/run_all_container`
# - Or invoke directly from any location:
#   `/home/cefect/LS/09_REPOS/04_TOOLS/floodsr/docs/user/notebooks/run_all_container`
#
# Requirements:
# - `docker` must be available on PATH from WSL.
# - The dev image must exist locally or be pullable.
# - The repo and notebook temp/data host paths must exist.
#
# Behavior:
# - Mounts only the repo plus the notebook temp/data paths needed for docs runs.
# - Runs `docs/user/notebooks/run_all.sh` inside the container so each notebook
#   shim still stages execution in temp, tutorial 3 still reuses `/_cache`, and
#   the refreshed notebooks are copied back into `docs/user/notebooks/`.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
image_name="${FLOODSR_NOTEBOOK_IMAGE:-cefect/floodsr:miniforge-dev-v1.0}"
tmpdir_host="${FLOODSR_TMPDIR_HOST:-/home/cefect/LS/10_IO/2407_FHIMP/tmp}"
io_host="${FLOODSR_IO_HOST:-/home/cefect/LS/10_IO/2407_FHIMP}"
cpus="${FLOODSR_NOTEBOOK_CPUS:-6}"
memory="${FLOODSR_NOTEBOOK_MEMORY:-12g}"

if ! command -v docker >/dev/null 2>&1; then
    echo "[run_all_container] error: 'docker' was not found on PATH" >&2
    exit 1
fi

# Fail early on the minimal host paths needed for notebook execution.
for path in "${repo_root}" "${tmpdir_host}" "${io_host}"; do
    if [ ! -e "${path}" ]; then
        echo "[run_all_container] error: required host path is missing" >&2
        echo "    ${path}" >&2
        exit 1
    fi
done

# Run the notebook refresh in a dedicated one-off container so this script does
# not attach to or mutate an already-running devcontainer session.
echo "[run_all_container] repo_root=${repo_root}"
echo "[run_all_container] image_name=${image_name}"
echo "[run_all_container] tmpdir_host=${tmpdir_host}"
echo "[run_all_container] io_host=${io_host}"
echo "[run_all_container] cpus=${cpus}"
echo "[run_all_container] memory=${memory}"
echo "[run_all_container] launching docs/user/notebooks/run_all.sh in a dedicated container"
docker run --rm -t \
    --entrypoint /bin/bash \
    --user 1000:1000 \
    --workdir /workspace \
    --pid host \
    --cpus "${cpus}" \
    --memory "${memory}" \
    -e TMPDIR=/home/cefect/LS/10_IO/2407_FHIMP/tmp \
    -e PYTHONPATH=/workspace \
    -v "${repo_root}:/workspace:delegated" \
    -v "${io_host}:/home/cefect/LS/10_IO/2407_FHIMP:delegated" \
    -v /etc/localtime:/etc/localtime:ro \
    -v /etc/timezone:/etc/timezone:ro \
    "${image_name}" \
    -lc "bash /workspace/docs/user/notebooks/run_all.sh"
