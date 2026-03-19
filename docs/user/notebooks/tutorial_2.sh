#!/usr/bin/env bash
set -euo pipefail

# Resolve notebook, staging, and cache paths up front so execution is reproducible.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
notebook_fp="${script_dir}/tutorial_2.ipynb"
cache_dir="${FLOODSR_NOTEBOOK_CACHE_DIR:-${repo_root}/_cache/notebook_tmp/tutorial_2}"
run_dir="${cache_dir}/run"
tmp_dir="${cache_dir}/tmp"
timeout_s="${FLOODSR_NOTEBOOK_TIMEOUT:-1200}"

# Reuse the same local package and cache paths as the notebook pytest workflow.
rm -rf "${run_dir}" "${tmp_dir}"
mkdir -p "${cache_dir}" "${run_dir}" "${tmp_dir}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${run_dir}:${PATH}"
export FLOODSR_NOTEBOOK_CACHE_DIR="${cache_dir}"
export HRDEM_CACHE_DIR="${cache_dir}"
export TMPDIR="${tmp_dir}"
export TEMP="${tmp_dir}"
export TMP="${tmp_dir}"

echo "[tutorial_2] notebook_fp=${notebook_fp}"
echo "[tutorial_2] repo_root=${repo_root}"
echo "[tutorial_2] cache_dir=${cache_dir}"
echo "[tutorial_2] run_dir=${run_dir}"
echo "[tutorial_2] python=$(python -c 'import sys; print(sys.executable)')"
echo "[tutorial_2] jupyter=$(python -m jupyter --version | tr '\n' ' ' )"
echo "[tutorial_2] staging notebook execution in ${run_dir}"

cp "${notebook_fp}" "${run_dir}/"

# Provide a local `floodsr` command inside the staged run directory so the
# user-facing notebook cells can keep the simple `!floodsr ...` form while the
# runner still executes against the repo source tree without requiring install steps.
cat > "${run_dir}/floodsr" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${repo_root}\${PYTHONPATH:+:\${PYTHONPATH}}"
argv=("\$@")
printf '[tutorial_2 floodsr] argv='
printf ' %q' python -m floodsr.cli "\${argv[@]}"
printf '\n'
python -m floodsr.cli "\${argv[@]}"
EOF
chmod +x "${run_dir}/floodsr"
cd "${run_dir}"
time python -m jupyter nbconvert \
    --debug \
    --to notebook \
    --execute \
    --inplace \
    "--ExecutePreprocessor.timeout=${timeout_s}" \
    "$(basename "${notebook_fp}")"

cp "${run_dir}/$(basename "${notebook_fp}")" "${notebook_fp}"
find "${run_dir}" -maxdepth 1 -type f ! -name "$(basename "${notebook_fp}")" -print | sed 's#^#[tutorial_2] staged output: #'
rm -rf "${run_dir}" "${tmp_dir}"
echo "[tutorial_2] refreshed source notebook without leaving side files in ${script_dir}"
