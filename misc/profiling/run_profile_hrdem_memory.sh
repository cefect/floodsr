#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace"
OUTPUT_DIR="${ROOT_DIR}/misc/profiling/output"
CASES=("non_windowed" "windowed_w512" "windowed_w256")
QUICK_CHECK="${PROFILE_HRDEM_QUICK_CHECK:-0}"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"
export PYTHONUNBUFFERED=1

echo "[INFO] HRDEM profiling entrypoint"
echo "[INFO] root_dir=${ROOT_DIR}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] quick_check=${QUICK_CHECK}"
echo "[INFO] cases=${CASES[*]}"

run_case() {
    local case_id="$1"
    local log_fp="${OUTPUT_DIR}/${case_id}.log"

    rm -f "${log_fp}"
    echo "[INFO] starting case=${case_id}"
    echo "[INFO] case_log=${log_fp}"

    if [[ "${QUICK_CHECK}" == "1" ]]; then
        CASE_ID="${case_id}" OUTPUT_DIR="${OUTPUT_DIR}" python - <<'PY'
import json, os

from misc.profiling.profile_hrdem_memory import CASE_D, DEPTH_LR_FP

case_id = os.environ["CASE_ID"]
print(
    json.dumps(
        {
            "status": "quick_check",
            "case_id": case_id,
            "depth_lr_fp": str(DEPTH_LR_FP),
            "output_dir": os.environ["OUTPUT_DIR"],
            "case_cfg": CASE_D[case_id],
        },
        indent=2,
    )
)
PY
    else
        CASE_ID="${case_id}" OUTPUT_DIR="${OUTPUT_DIR}" python - <<'PY'
import os

from misc.profiling.profile_hrdem_memory import main_profile_hrdem_memory

main_profile_hrdem_memory(
    os.environ["CASE_ID"],
    output_dir=os.environ["OUTPUT_DIR"],
)
PY
    fi

    echo "[INFO] finished case=${case_id}"
}

for case_id in "${CASES[@]}"; do
    # Each Python invocation blocks, so the cases run strictly in sequence.
    run_case "${case_id}"
done

if [[ "${QUICK_CHECK}" == "1" ]]; then
    echo "[INFO] quick check complete; summary build skipped"
    exit 0
fi

echo "[INFO] building combined summary"
OUTPUT_DIR="${OUTPUT_DIR}" python - <<'PY'
import os

from misc.profiling.profile_hrdem_memory import main_write_profile_summary

main_write_profile_summary(output_dir=os.environ["OUTPUT_DIR"])
PY
echo "[INFO] profiling sequence complete"
