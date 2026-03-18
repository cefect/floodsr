#!/usr/bin/env bash
set -euo pipefail

# Resolve the repo root from this script location so callers can run from anywhere.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
image="${FLOODSR_NOTEBOOK_IMAGE:-cefect/floodsr:tutorial-notebooks-v0.1}"
notebooks_dir="${FLOODSR_NOTEBOOKS_DIR:-docs/user/notebooks}"
pattern="${FLOODSR_NOTEBOOK_PATTERN:-*.ipynb}"
timeout="${FLOODSR_NOTEBOOK_TIMEOUT:-600}"

# Launch the prebuilt tutorial notebook container with the repo mounted at /workspace.
if [[ "${FLOODSR_NOTEBOOKS_IN_CONTAINER:-0}" != "1" ]]; then
    exec docker run --rm \
        --entrypoint /bin/bash \
        -e FLOODSR_NOTEBOOKS_IN_CONTAINER=1 \
        -e FLOODSR_NOTEBOOKS_DIR="${notebooks_dir}" \
        -e FLOODSR_NOTEBOOK_PATTERN="${pattern}" \
        -e FLOODSR_NOTEBOOK_TIMEOUT="${timeout}" \
        -v "${repo_root}:/workspace" \
        -w /workspace \
        "${image}" \
        -lc "bash docs/user/scripts/run_notebooks.sh"
fi

# Collect notebooks in a stable order before executing them in place.
mapfile -t notebook_fp_l < <(find "${notebooks_dir}" -maxdepth 1 -type f -name "${pattern}" | sort)
if [[ "${#notebook_fp_l[@]}" -eq 0 ]]; then
    echo "[run_notebooks] no notebooks matched pattern '${pattern}' in '${notebooks_dir}'" >&2
    exit 1
fi

# Execute each notebook in place so rendered outputs are stored in the source file.
for notebook_fp in "${notebook_fp_l[@]}"; do
    echo "[run_notebooks] executing ${notebook_fp}"
    jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        "--ExecutePreprocessor.timeout=${timeout}" \
        "${notebook_fp}"
done

# Remove downloaded and derived GeoTIFF artifacts so the notebook directory stays clean.
mapfile -t tif_fp_l < <(find "${notebooks_dir}" -maxdepth 1 -type f -name '*.tif' | sort)
if [[ "${#tif_fp_l[@]}" -gt 0 ]]; then
    printf '%s\0' "${tif_fp_l[@]}" | xargs -0 rm -f
    echo "[run_notebooks] removed ${#tif_fp_l[@]} tif file(s)"
fi

echo "[run_notebooks] executed ${#notebook_fp_l[@]} notebook(s)"
