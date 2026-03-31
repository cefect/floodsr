#!/usr/bin/env bash
set -euo pipefail

# Run tutorial 2 from a temp sandbox and copy the executed notebook back into the docs tree.
#
# Usage:
# - `conda run -n dev bash docs/user/notebooks/tutorial_2.sh`
#
# The runner mirrors the notebook pytest layout so the documented workflow runs
# against the in-repo source tree with disposable temp directories.
# A small staged `floodsr` wrapper preserves the tutorial's `!floodsr ...` cells.
#
# Environment overrides:
# - `FLOODSR_NOTEBOOK_STAGE_DIR` controls the per-notebook staging root.
# - `FLOODSR_NOTEBOOK_CACHE_DIR` remains as a backward-compatible alias.
# - `FLOODSR_NOTEBOOK_TIMEOUT` sets the nbconvert execution timeout in seconds.
#
# Outputs:
# - Overwrites `tutorial_2.ipynb` with the freshly executed notebook.
# - Removes the temporary run and tmp directories before exiting.
#
# Resolve notebook and staging paths up front so execution is reproducible.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
notebook_fp="${script_dir}/tutorial_2.ipynb"
stage_dir="${FLOODSR_NOTEBOOK_STAGE_DIR:-${FLOODSR_NOTEBOOK_CACHE_DIR:-}}"
cleanup_stage=0
if [ -z "${stage_dir}" ]; then
    stage_dir="$(mktemp -d -t floodsr-tutorial_2-XXXXXX)"
    cleanup_stage=1
fi
run_dir="${stage_dir}/run"
tmp_dir="${stage_dir}/tmp"
timeout_s="${FLOODSR_NOTEBOOK_TIMEOUT:-1200}"

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

echo "[tutorial_2] notebook_fp=${notebook_fp}"
echo "[tutorial_2] repo_root=${repo_root}"
echo "[tutorial_2] stage_dir=${stage_dir}"
echo "[tutorial_2] run_dir=${run_dir}"
echo "[tutorial_2] python=$(python -c 'import sys; print(sys.executable)')"
echo "[tutorial_2] jupyter=$(python -m jupyter --version | tr '\n' ' ' )"
echo "[tutorial_2] staging notebook execution in ${run_dir}"

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
echo "[tutorial_2] executing notebook"
time python -m jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    "--ExecutePreprocessor.timeout=${timeout_s}" \
    "$(basename "${notebook_fp}")"

cp "${run_dir}/$(basename "${notebook_fp}")" "${notebook_fp}"
echo "[tutorial_2] refreshed source notebook without leaving side files in ${script_dir}"
