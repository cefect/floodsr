#!/usr/bin/env bash
set -euo pipefail

# Run tutorial 1 from a temp sandbox and copy the executed notebook back into the docs tree.
#
# Usage:
# - `conda run -n dev bash docs/user/notebooks/tutorial_1.sh`
#
# The runner stages a temporary CLI shim plus writable temp directories so the
# notebook's `!floodsr ...` cells execute against the repo source tree without
# requiring a package install or leaving side files in the source directory.
#
# Environment overrides:
# - `FLOODSR_NOTEBOOK_STAGE_DIR` controls the per-notebook staging root.
# - `FLOODSR_NOTEBOOK_CACHE_DIR` remains as a backward-compatible alias.
# - `FLOODSR_NOTEBOOK_TIMEOUT` sets the nbconvert execution timeout in seconds.
#
# Outputs:
# - Overwrites `tutorial_1.ipynb` with the freshly executed notebook.
# - Removes the temporary run and tmp directories before exiting.
#
# Resolve notebook and staging paths up front so execution is reproducible.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
notebook_fp="${script_dir}/tutorial_1.ipynb"
stage_dir="${FLOODSR_NOTEBOOK_STAGE_DIR:-${FLOODSR_NOTEBOOK_CACHE_DIR:-}}"
cleanup_stage=0
if [ -z "${stage_dir}" ]; then
    stage_dir="$(mktemp -d -t floodsr-tutorial_1-XXXXXX)"
    cleanup_stage=1
fi
run_dir="${stage_dir}/run"
tmp_dir="${stage_dir}/tmp"
timeout_s="${FLOODSR_NOTEBOOK_TIMEOUT:-600}"

# Fail early if the notebook runtime is unavailable in the active environment.
if ! command -v jupyter >/dev/null 2>&1; then
    echo "[tutorial_1] error: 'jupyter' was not found on PATH" >&2
    echo "[tutorial_1] hint: run this from the notebook-capable environment, e.g. 'conda run -n dev bash docs/user/notebooks/tutorial_1.sh'" >&2
    exit 1
fi

# Keep notebook side files inside the temp-backed stage directory and remove
# them even when notebook execution fails.
trap 'if [ "${cleanup_stage}" -eq 1 ]; then rm -rf "${stage_dir}"; else rm -rf "${run_dir}" "${tmp_dir}"; fi' EXIT
mkdir -p "${stage_dir}" "${run_dir}" "${tmp_dir}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${run_dir}:${PATH}"
export FLOODSR_NOTEBOOK_STAGE_DIR="${stage_dir}"
export FLOODSR_NOTEBOOK_CACHE_DIR="${stage_dir}"
export HRDEM_CACHE_DIR="${stage_dir}"
export TMPDIR="${tmp_dir}"
export TEMP="${tmp_dir}"
export TMP="${tmp_dir}"

echo "[tutorial_1] notebook_fp=${notebook_fp}"
echo "[tutorial_1] repo_root=${repo_root}"
echo "[tutorial_1] stage_dir=${stage_dir}"
echo "[tutorial_1] run_dir=${run_dir}"
echo "[tutorial_1] python=$(python -c 'import sys; print(sys.executable)')"
echo "[tutorial_1] jupyter=$(python -m jupyter --version | tr '\n' ' ' )"
echo "[tutorial_1] staging notebook execution in ${run_dir}"

cp "${notebook_fp}" "${run_dir}/"

# Provide a local `floodsr` command inside the staged run directory so the
# notebook can keep the simple `!floodsr ...` examples shown to users while the
# runner still dispatches to `python -m floodsr.cli` from the repo checkout.
cat > "${run_dir}/floodsr" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${repo_root}\${PYTHONPATH:+:\${PYTHONPATH}}"
python -m floodsr.cli "\$@"
EOF
chmod +x "${run_dir}/floodsr"
cd "${run_dir}"
echo "[tutorial_1] executing notebook"
time python -m jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    "--ExecutePreprocessor.timeout=${timeout_s}" \
    "$(basename "${notebook_fp}")"

cp "${run_dir}/$(basename "${notebook_fp}")" "${notebook_fp}"
echo "[tutorial_1] refreshed source notebook without leaving side files in ${script_dir}"
