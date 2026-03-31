#!/usr/bin/env bash
set -euo pipefail

# Run tutorial 3 from a temp sandbox and copy the executed notebook back into the docs tree.
#
# Usage:
# - `conda run -n dev bash docs/user/notebooks/tutorial_3.sh`
#
# This tutorial can reuse a shared project cache for larger intermediate
# products while still staging notebook execution in a disposable per-run
# temp directory. Like the
# other runners, it injects a local `floodsr` shim so the documented CLI cells
# run directly against the repo source tree without install steps.
#
# Environment overrides:
# - `FLOODSR_SHARED_CACHE_DIR` controls the shared project cache location.
# - `FLOODSR_NOTEBOOK_STAGE_DIR` controls the disposable staging root.
# - `FLOODSR_NOTEBOOK_CACHE_DIR` remains as a backward-compatible alias.
# - `FLOODSR_NOTEBOOK_TIMEOUT` sets the nbconvert execution timeout in seconds.
#
# Outputs:
# - Overwrites `tutorial_3.ipynb` with the freshly executed notebook.
# - Removes the temporary run and tmp directories before exiting.
#
# Resolve notebook, staging, and cache paths up front so execution is reproducible.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
notebook_fp="${script_dir}/tutorial_3.ipynb"
shared_cache_dir="${FLOODSR_SHARED_CACHE_DIR:-/home/cefect/LS/09_REPOS/04_TOOLS/floodsr/_cache}"
stage_dir="${FLOODSR_NOTEBOOK_STAGE_DIR:-${FLOODSR_NOTEBOOK_CACHE_DIR:-}}"
cleanup_stage=0
if [ -z "${stage_dir}" ]; then
    stage_dir="$(mktemp -d -t floodsr-tutorial_3-XXXXXX)"
    cleanup_stage=1
fi
run_dir="${stage_dir}/run"
tmp_dir="${stage_dir}/tmp"
timeout_s="${FLOODSR_NOTEBOOK_TIMEOUT:-3600}"

# Fail early if the notebook runtime is unavailable in the active environment.
if ! command -v jupyter >/dev/null 2>&1; then
    echo "[tutorial_3] error: 'jupyter' was not found on PATH" >&2
    echo "[tutorial_3] hint: run this from the notebook-capable environment, e.g. 'conda run -n dev bash docs/user/notebooks/tutorial_3.sh'" >&2
    exit 1
fi

# Keep notebook side files inside the temp-backed stage directory, but let the
# notebook point heavy HRDEM/model reuse at the shared project cache.
trap 'if [ "${cleanup_stage}" -eq 1 ]; then rm -rf "${stage_dir}"; else rm -rf "${run_dir}" "${tmp_dir}"; fi' EXIT
mkdir -p "${shared_cache_dir}" "${stage_dir}" "${run_dir}" "${tmp_dir}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${run_dir}:${PATH}"
export FLOODSR_SHARED_CACHE_DIR="${shared_cache_dir}"
export FLOODSR_NOTEBOOK_STAGE_DIR="${stage_dir}"
export FLOODSR_NOTEBOOK_CACHE_DIR="${stage_dir}"
export TMPDIR="${tmp_dir}"
export TEMP="${tmp_dir}"
export TMP="${tmp_dir}"

echo "[tutorial_3] notebook_fp=${notebook_fp}"
echo "[tutorial_3] repo_root=${repo_root}"
echo "[tutorial_3] shared_cache_dir=${shared_cache_dir}"
echo "[tutorial_3] stage_dir=${stage_dir}"
echo "[tutorial_3] run_dir=${run_dir}"
echo "[tutorial_3] python=$(python -c 'import sys; print(sys.executable)')"
echo "[tutorial_3] jupyter=$(python -m jupyter --version | tr '\n' ' ' )"
echo "[tutorial_3] staging notebook execution in ${run_dir}"
echo "[tutorial_3] note: the HRDEM fetch and tohr cells can take a while"

cp "${notebook_fp}" "${run_dir}/"

# Provide a local `floodsr` command inside the staged run directory so notebook
# CLI cells keep the documented command form while still executing from the
# repo checkout through `python -m floodsr.cli`.
cat > "${run_dir}/floodsr" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${repo_root}\${PYTHONPATH:+:\${PYTHONPATH}}"
python -m floodsr.cli "\$@"
EOF
chmod +x "${run_dir}/floodsr"

cd "${run_dir}"
echo "[tutorial_3] executing notebook"
time python -m jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    "--ExecutePreprocessor.timeout=${timeout_s}" \
    "$(basename "${notebook_fp}")"

cp "${run_dir}/$(basename "${notebook_fp}")" "${notebook_fp}"
echo "[tutorial_3] refreshed source notebook without leaving side files in ${script_dir}"
