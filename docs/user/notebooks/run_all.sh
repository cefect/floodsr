#!/usr/bin/env bash
set -euo pipefail

# Run all tutorial notebook refresh shims from inside the notebook-capable
# devcontainer environment.
#
# Usage:
# - `bash docs/user/notebooks/run_all.sh` from an already-active notebook-capable shell
# - `conda run -n dev bash docs/user/notebooks/run_all.sh`
#
# Progress notes:
# - starts all three notebook shims in parallel
# - prints each PID after launch
# - reports completion per tutorial as waits resolve
# - returns non-zero if any tutorial fails

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

# Fail early if the expected conda runtime is unavailable.
if ! command -v conda >/dev/null 2>&1; then
    echo "[run_all] error: 'conda' was not found on PATH" >&2
    exit 1
fi

echo "[run_all] repo_root=${repo_root}"
echo "[run_all] launching tutorial refresh in parallel"

cd "${repo_root}"
conda run -n dev bash docs/user/notebooks/tutorial_1.sh &
pid_1=$!
echo "[run_all] started tutorial_1.sh pid=${pid_1}"
conda run -n dev bash docs/user/notebooks/tutorial_2.sh &
pid_2=$!
echo "[run_all] started tutorial_2.sh pid=${pid_2}"
conda run -n dev bash docs/user/notebooks/tutorial_3.sh &
pid_3=$!
echo "[run_all] started tutorial_3.sh pid=${pid_3}"

status=0
for pair in "tutorial_1:${pid_1}" "tutorial_2:${pid_2}" "tutorial_3:${pid_3}"; do
    name="${pair%%:*}"
    pid="${pair##*:}"
    echo "[run_all] waiting for ${name} pid=${pid}"
    if ! wait "${pid}"; then
        echo "[run_all] ${name} failed"
        status=1
    else
        echo "[run_all] ${name} completed"
    fi
done

if [ "${status}" -eq 0 ]; then
    echo "[run_all] all notebook refreshes completed"
else
    echo "[run_all] one or more notebook refreshes failed"
fi

exit "${status}"
