#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

# Runs main_local.sh once per existing subfolder in <repo_root>/results.
# Uses the folder name as RUN_NAME so each run writes into that folder.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MAIN="${REPO_ROOT}/tools/main_local.sh"   # <-- adjust if your main_local.sh lives elsewhere
RESULTS_DIR="${REPO_ROOT}/results"

if [[ ! -x "${MAIN}" ]]; then
  echo "[run_all_results.sh] ERROR: main_local.sh not found/executable at: ${MAIN}"
  echo "[run_all_results.sh] Fix MAIN path in this script."
  exit 1
fi

if [[ ! -d "${RESULTS_DIR}" ]]; then
  echo "[run_all_results.sh] ERROR: results dir not found: ${RESULTS_DIR}"
  exit 1
fi

mapfile -t RUNS < <(find "${RESULTS_DIR}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)

if (( ${#RUNS[@]} == 0 )); then
  echo "[run_all_results.sh] No subfolders found in ${RESULTS_DIR}"
  exit 0
fi

echo "[run_all_results.sh] Found ${#RUNS[@]} runs:"
printf "  - %s\n" "${RUNS[@]}"

FAIL_LOG="${RESULTS_DIR}/run_all_failures.log"
: > "${FAIL_LOG}"

run_one() {
  local run="$1"

  echo "============================================================"
  echo "[run_all_results.sh] Running: ${run}"
  echo "============================================================"

  local overview_dir="${RESULTS_DIR}/${run}/overview"
  local stdout_log="${RESULTS_DIR}/${run}/run_all_stdout.log"

  if [[ -d "${overview_dir}" ]]; then
    find "${overview_dir}" -mindepth 1 ! -name "*.pth" -exec rm -rf {} +
  fi

  # Run and tee output
  bash "${MAIN}" "${run}" | tee "${stdout_log}"
}

for run in "${RUNS[@]}"; do
  # "try/except": run in an if; on failure, record + continue.
  if run_one "${run}"; then
    echo "[run_all_results.sh] OK: ${run}"
  else
    rc=$?
    echo "[run_all_results.sh] FAIL (exit ${rc}): ${run}"
    echo "$(date -Iseconds) | ${run} | exit ${rc}" >> "${FAIL_LOG}"
    continue
  fi
done

echo "[run_all_results.sh] DONE"
echo "[run_all_results.sh] Failures log: ${FAIL_LOG}"