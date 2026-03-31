#!/usr/bin/env bash
set -euo pipefail

# Run tutorial 1 from cache and copy the executed notebook back into the docs tree.
#
# Usage:
# - `conda run -n dev bash docs/user/notebooks/tutorial_1.sh`
#
# The runner stages a temporary CLI shim plus writable cache directories so the
# notebook's `!floodsr ...` cells execute against the repo source tree without
# requiring a package install or leaving side files in the source directory.
#
# Environment overrides:
# - `FLOODSR_NOTEBOOK_CACHE_DIR` controls the per-notebook staging/cache root.
# - `FLOODSR_NOTEBOOK_TIMEOUT` sets the nbconvert execution timeout in seconds.
#
# Outputs:
# - Overwrites `tutorial_1.ipynb` with the freshly executed notebook.
# - Removes the temporary run and tmp directories before exiting.
#
# Resolve notebook, staging, and cache paths up front so execution is reproducible.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
notebook_fp="${script_dir}/tutorial_1.ipynb"
cache_dir="${FLOODSR_NOTEBOOK_CACHE_DIR:-${repo_root}/_cache/notebook_tmp/tutorial_1}"
run_dir="${cache_dir}/run"
tmp_dir="${cache_dir}/tmp"
timeout_s="${FLOODSR_NOTEBOOK_TIMEOUT:-600}"

# Reuse the same local package and cache paths as the notebook pytest workflow
# so docs runs and tests exercise the same import and cache layout.
rm -rf "${run_dir}" "${tmp_dir}"
mkdir -p "${cache_dir}" "${run_dir}" "${tmp_dir}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${run_dir}:${PATH}"
export FLOODSR_NOTEBOOK_CACHE_DIR="${cache_dir}"
export HRDEM_CACHE_DIR="${cache_dir}"
export TMPDIR="${tmp_dir}"
export TEMP="${tmp_dir}"
export TMP="${tmp_dir}"

echo "[tutorial_1] notebook_fp=${notebook_fp}"
echo "[tutorial_1] repo_root=${repo_root}"
echo "[tutorial_1] cache_dir=${cache_dir}"
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
rm -rf "${run_dir}" "${tmp_dir}"
echo "[tutorial_1] refreshed source notebook without leaving side files in ${script_dir}"
