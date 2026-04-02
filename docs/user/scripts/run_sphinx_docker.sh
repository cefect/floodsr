#!/usr/bin/env bash

# Run the docs Sphinx pipeline from the host with the docs image defaults inlined.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../../.." && pwd)"
french=0

# Keep the CLI narrow: English by default, French only when explicitly requested.
if [[ "${1:-}" == "--french" ]]; then
  french=1
  shift
fi

if [[ $# -ne 0 ]]; then
  printf "Usage: %s [--french]\n" "$(basename "${BASH_SOURCE[0]}")" >&2
  exit 2
fi

docker_args=(
  run
  --rm
  --user "1000:1000"
  --workdir /workspace
  --env PYTHONPATH=/workspace
  --volume "${repo_dir}:/workspace:delegated"
  --volume /home/cefect/LS/10_IO:/home/cefect/LS/10_IO:rw
  --volume /home/cefect/.config:/home/cefect/.config:rw
  --volume /home/cefect/.codex:/home/cefect/.codex:rw
  --volume /home/cefect/.ssh:/home/cefect/.ssh:ro
  --volume /etc/localtime:/etc/localtime:ro
  --volume /etc/timezone:/etc/timezone:ro
  "cefect/floodsr-docs:dev-v0.1"
  bash /workspace/docs/user/scripts/_run_sphinx_inside.sh
)

# Forward the optional language switch to the container-side build shim.
if [[ "${french}" -eq 1 ]]; then
  docker_args+=(--french)
fi

exec docker "${docker_args[@]}"
