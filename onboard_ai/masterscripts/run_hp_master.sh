#!/usr/bin/env bash
set -euo pipefail

# ===================== USER SETTINGS =====================

# Source list (one run name per line, no .yml)
SRC_FILE="${1:-$HOME/generated_configs.txt}"

# Only use these repos
FOLDERS=(1 2 3 4 5 6 7 8 9)
REPO_PREFIX="$HOME/DEIMv2-main"

# Poll interval in seconds
POLL_SEC=30

# Log you can tail
MASTER_LOG="$HOME/master_log.txt"

# Configs must exist here (relative to repo root)
CFG_DIR_REL="configs/6_refine"

# Conda env
CONDA_ENV="deimv2"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"

# Tracking dir
MASTER_DIR="$HOME/hp_master"

# Central results dir (SUCCESS only)
MASTER_RESULTS="$HOME/MASTER_RESULTS"

# Central results dir (FAILED / interrupted / debug)
MASTER_RESULTS_DEBUG="$HOME/MASTER_RESULTS_debug"

# Queue files
WAITING_FILE="$HOME/generated_configs_WAITING.txt"
TASKED_FILE="$HOME/generated_configs_TASKED.txt"
DONE_FILE="$HOME/generated_configs_DONE.txt"

# Kill command file (ONE command: kills master + all descendants)
KILL_FILE="$HOME/kill_master.txt"

# If 1: treat repos as busy when they have a foreign run (manual run)
DETECT_FOREIGN_BUSY="${DETECT_FOREIGN_BUSY:-1}"

# =================== DO NOT EDIT BELOW ===================

ts() { date +"%Y-%m-%d %H:%M:%S"; }

# IMPORTANT: never write logs to stdout (stdout is used for command substitution outputs)
log() {
  local msg="[$(ts)] $*"
  echo "$msg" >> "$MASTER_LOG"
  echo "$msg" >&2
}

die() { log "ERROR: $*"; exit 1; }

require_file() { [[ -f "$1" ]] || die "Missing file: $1"; }
require_dir() { [[ -d "$1" ]] || die "Missing dir: $1"; }

pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" >/dev/null 2>&1
}

pid_file_for() { echo "$MASTER_DIR/pid_$1.txt"; }
run_file_for() { echo "$MASTER_DIR/run_$1.txt"; }
rc_file_for()  { echo "$MASTER_DIR/rc_$1.txt"; }

clear_instance() {
  local inst="$1"
  rm -f "$(pid_file_for "$inst")" "$(run_file_for "$inst")" "$(rc_file_for "$inst")" >/dev/null 2>&1 || true
}

norm_list() {
  awk '
    {
      gsub(/\r/, "", $0)
      sub(/^[[:space:]]+/, "", $0)
      sub(/[[:space:]]+$/, "", $0)
      if ($0 ~ /^[[:space:]]*$/) next
      if ($0 ~ /^[[:space:]]*#/) next
      print $0
    }
  ' "$1"
}

set_run_name_in_main() {
  local main_sh="$1"
  local run_name="$2"
  require_file "$main_sh"
  grep -qE '^RUN_NAME=' "$main_sh" || die "RUN_NAME line not found in $main_sh"
  perl -0777 -i -pe "s/^RUN_NAME=\"[^\"]*\"/RUN_NAME=\"$run_name\"/m" "$main_sh"
}

# Only skip if MASTER_RESULTS contains it (SUCCESS only)
is_done_master_results() {
  local run="$1"
  [[ -d "${MASTER_RESULTS}/${run}" ]]
}

# ---------- DONE based on MASTER_RESULTS (SUCCESS only) ----------
rebuild_done_from_master_results() {
  local tmp="$MASTER_DIR/.done.tmp"
  : > "$tmp"

  if [[ -d "$MASTER_RESULTS" ]]; then
    find "$MASTER_RESULTS" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" \
      | sed -E 's/_[0-9]{9,}$//' \
      | sort -u > "$tmp"
  fi

  mv -f "$tmp" "$DONE_FILE"
  log "[queue] rebuilt DONE from MASTER_RESULTS -> $DONE_FILE"
}

# ---------- WAITING from SRC minus DONE ----------
rebuild_waiting_from_src_minus_done() {
  local src_tmp="$MASTER_DIR/.src_norm.tmp"
  local done_tmp="$MASTER_DIR/.done_norm.tmp"
  local out_tmp="$MASTER_DIR/.waiting.tmp"

  norm_list "$SRC_FILE" | sort -u > "$src_tmp"
  if [[ -f "$DONE_FILE" ]]; then
    norm_list "$DONE_FILE" | sort -u > "$done_tmp"
  else
    : > "$done_tmp"
  fi

  comm -23 "$src_tmp" "$done_tmp" > "$out_tmp"
  mv -f "$out_tmp" "$WAITING_FILE"

  rm -f "$src_tmp" "$done_tmp" >/dev/null 2>&1 || true
  log "[queue] rebuilt WAITING from SRC\\DONE -> $WAITING_FILE"
}

clear_tasked_on_startup() {
  : > "$TASKED_FILE"
  log "[queue] cleared TASKED -> $TASKED_FILE"
}

# ---------- Atomic moves between files ----------
queue_lock="$MASTER_DIR/.queue.lock"

move_first_line_waiting_to_tasked() {
  local tmpw="$MASTER_DIR/.waiting.tmp"
  local picked=""

  if command -v flock >/dev/null 2>&1; then
    (
      flock -x 9
      mapfile -t lines < <(norm_list "$WAITING_FILE" || true)
      [[ "${#lines[@]}" -gt 0 ]] || { echo ""; exit 0; }
      picked="${lines[0]}"

      awk -v c="$picked" '
        BEGIN{removed=0}
        {
          line=$0
          gsub(/\r/, "", line)
          sub(/^[[:space:]]+/, "", line)
          sub(/[[:space:]]+$/, "", line)
          if (line ~ /^[[:space:]]*$/) next
          if (line ~ /^[[:space:]]*#/) next
          if (removed==0 && line==c) { removed=1; next }
          print line
        }
      ' "$WAITING_FILE" > "$tmpw"
      mv -f "$tmpw" "$WAITING_FILE"

      echo "$picked" >> "$TASKED_FILE"
      echo "$picked"
    ) 9>"$queue_lock"
    return 0
  fi

  mapfile -t lines < <(norm_list "$WAITING_FILE" || true)
  [[ "${#lines[@]}" -gt 0 ]] || { echo ""; return 0; }
  picked="${lines[0]}"
  awk -v c="$picked" '
    BEGIN{removed=0}
    {
      line=$0
      gsub(/\r/, "", line)
      sub(/^[[:space:]]+/, "", line)
      sub(/[[:space:]]+$/, "", line)
      if (line ~ /^[[:space:]]*$/) next
      if (line ~ /^[[:space:]]*#/) next
      if (removed==0 && line==c) { removed=1; next }
      print line
    }
  ' "$WAITING_FILE" > "$tmpw"
  mv -f "$tmpw" "$WAITING_FILE"
  echo "$picked" >> "$TASKED_FILE"
  echo "$picked"
}

remove_from_tasked() {
  local run="$1"
  local tmpt="$MASTER_DIR/.tasked.tmp"
  if [[ -f "$TASKED_FILE" ]]; then
    awk -v c="$run" '
      {
        line=$0
        gsub(/\r/,"",line)
        sub(/^[[:space:]]+/,"",line)
        sub(/[[:space:]]+$/,"",line)
        if (line==c) next
        if (line ~ /^[[:space:]]*$/) next
        if (line ~ /^[[:space:]]*#/) next
        print line
      }
    ' "$TASKED_FILE" > "$tmpt" || true
    mv -f "$tmpt" "$TASKED_FILE"
  fi
}

add_to_done() {
  local run="$1"
  local tmpd="$MASTER_DIR/.done.tmp"
  if [[ -f "$DONE_FILE" ]]; then
    norm_list "$DONE_FILE" > "$tmpd"
  else
    : > "$tmpd"
  fi
  if ! grep -Fxq "$run" "$tmpd"; then
    echo "$run" >> "$tmpd"
  fi
  sort -u "$tmpd" > "${tmpd}.s"
  mv -f "${tmpd}.s" "$DONE_FILE"
  rm -f "$tmpd" >/dev/null 2>&1 || true
}

# ---------- results move ----------
move_results_to_central() {
  local inst="$1"
  local run="$2"
  local rc="$3"

  local repo="${REPO_PREFIX}_${inst}"
  local src="${repo}/results/${run}"

  local central=""
  if [[ "$rc" -eq 0 ]]; then
    central="$MASTER_RESULTS"
  else
    central="$MASTER_RESULTS_DEBUG"
  fi

  if [[ ! -d "$src" ]]; then
    log "[inst=${inst}] No results folder found for ${run} (skipping move)"
    return 1
  fi

  mkdir -p "$central"

  local dst="${central}/${run}"
  if [[ -d "$dst" ]]; then
    log "[inst=${inst}] WARNING: destination already exists for ${run} in $(basename "$central"), renaming"
    dst="${central}/${run}_$(date +%s)"
  fi

  log "[inst=${inst}] Moving results ${src} -> ${dst} (rc=${rc})"
  mv "$src" "$dst"
  return 0
}

# ---------- foreign busy detection ----------
is_instance_busy_foreign() {
  local inst="$1"
  [[ "$DETECT_FOREIGN_BUSY" == "1" ]] || return 1

  local repo="${REPO_PREFIX}_${inst}"
  [[ -d "$repo" ]] || return 1

  local pid=""
  pid="$(
    ps -eo pid=,args= \
      | awk -v repo="$repo" '
          $0 ~ repo && ($0 ~ /python/ || $0 ~ /bash/) && ($0 ~ /train_crossval_deimv2\.py/ || $0 ~ /train_final\.py/ || $0 ~ /dump_predictions_all\.py/ || $0 ~ /evaluate_models\.py/ || $0 ~ /tools\/main\.sh/ ) {print $1; exit}
        ' || true
  )"

  if [[ -n "$pid" ]] && pid_alive "$pid"; then
    log "[inst=${inst}] FOREIGN BUSY detected: pid=${pid} repo=${repo}"
    return 0
  fi
  return 1
}

# ---------- "killmaster" one-liner ----------
update_kill_file() {
  local tmp="$MASTER_DIR/.kill.tmp"
  local master_pid="$$"

  {
    echo "# Auto-generated by run_hp_master.sh"
    echo "# Updated: $(ts)"
    echo "# One command: kill this master + ALL descendants (only this process tree)."
    echo "bash -lc 'MPID=${master_pid}; pkill -TERM -P \$MPID 2>/dev/null || true; kill -TERM \$MPID 2>/dev/null || true; sleep 2; pkill -KILL -P \$MPID 2>/dev/null || true; kill -KILL \$MPID 2>/dev/null || true'"
  } > "$tmp"

  mv -f "$tmp" "$KILL_FILE"
}

# ---------- start job ----------
start_job_on_instance() {
  local inst="$1"
  local run_name="$2"
  local repo="${REPO_PREFIX}_${inst}"
  local main_sh="$repo/tools/main.sh"
  local cfg="$repo/${CFG_DIR_REL}/${run_name}.yml"

  require_dir "$repo"
  require_file "$main_sh"

  if is_instance_busy_foreign "$inst"; then
    log "[inst=${inst}] SKIP scheduling (foreign busy)"
    return 3
  fi

  if is_done_master_results "$run_name"; then
    log "[inst=${inst}] SKIP run=${run_name} (already exists in MASTER_RESULTS)"
    return 2
  fi

  [[ -f "$cfg" ]] || { log "[inst=${inst}] FAIL run=${run_name} (missing config: ${cfg})"; return 1; }

  set_run_name_in_main "$main_sh" "$run_name"

  log "[inst=${inst}] START run=${run_name} repo=${repo}"
  echo "$run_name" > "$(run_file_for "$inst")"

  (
    cd "$repo"
    nohup bash -lc "
      set -euo pipefail
      source \"${CONDA_SH}\"
      conda activate \"${CONDA_ENV}\"
      exec bash tools/main.sh
    " > nohup_train.log 2>&1 &

    echo $! > "$(pid_file_for "$inst")"
    disown || true
  )

  local pid
  pid="$(cat "$(pid_file_for "$inst")" 2>/dev/null || true)"
  log "[inst=${inst}] PID=${pid} (slot busy until this PID exits)"
  update_kill_file
  return 0
}

find_free_instance() {
  local inst pf pid
  for inst in "${FOLDERS[@]}"; do
    if is_instance_busy_foreign "$inst"; then
      continue
    fi

    pf="$(pid_file_for "$inst")"
    if [[ ! -f "$pf" ]]; then
      printf '%s\n' "$inst"
      return 0
    fi
    pid="$(cat "$pf" 2>/dev/null || true)"
    if ! pid_alive "$pid"; then
      printf '%s\n' "$inst"
      return 0
    fi
  done
  echo ""
}

# ---------- cleanup on interrupt ----------
DRAIN_ON_EXIT="${DRAIN_ON_EXIT:-1}"

drain_finished_instances_on_exit() {
  [[ "$DRAIN_ON_EXIT" == "1" ]] || return 0

  log "[exit] Draining finished instances (no waiting for running jobs)..."
  for inst in "${FOLDERS[@]}"; do
    local pf pid last_run rc
    pf="$(pid_file_for "$inst")"
    [[ -f "$pf" ]] || continue
    pid="$(cat "$pf" 2>/dev/null || true)"
    last_run="$(cat "$(run_file_for "$inst")" 2>/dev/null || true)"
    [[ -n "$pid" && -n "$last_run" ]] || continue

    if pid_alive "$pid"; then
      log "[exit][inst=${inst}] still running pid=${pid} run=${last_run} (leaving results in repo)"
      continue
    fi

    set +e
    wait "$pid"
    rc=$?
    set -e
    echo "$rc" > "$(rc_file_for "$inst")" || true

    log "[exit][inst=${inst}] finished pid=${pid} run=${last_run} rc=${rc} (moving accordingly)"
    move_results_to_central "$inst" "$last_run" "$rc" || true

    if [[ "$rc" -eq 0 ]]; then
      remove_from_tasked "$last_run"
      add_to_done "$last_run"
    else
      remove_from_tasked "$last_run"
    fi

    clear_instance "$inst"
  done
}

on_exit() {
  local ec=$?
  if [[ "$ec" -ne 0 ]]; then
    log "[exit] Master exiting with code ${ec} (likely interrupt or error)."
  else
    log "[exit] Master exiting normally."
  fi
  drain_finished_instances_on_exit || true
  update_kill_file || true
}
trap on_exit EXIT

# ===================== MAIN =====================

mkdir -p "$MASTER_DIR"
require_file "$SRC_FILE"
require_file "$CONDA_SH"

: > "$WAITING_FILE" || true
: > "$TASKED_FILE" || true
: > "$DONE_FILE" || true

log "==================== MASTER START ===================="
log "SRC_FILE=$SRC_FILE"
log "WAITING_FILE=$WAITING_FILE"
log "TASKED_FILE=$TASKED_FILE"
log "DONE_FILE=$DONE_FILE"
log "FOLDERS=${FOLDERS[*]}"
log "REPO_PREFIX=$REPO_PREFIX"
log "CFG_DIR_REL=$CFG_DIR_REL"
log "CONDA_ENV=$CONDA_ENV"
log "CONDA_SH=$CONDA_SH"
log "MASTER_DIR=$MASTER_DIR"
log "MASTER_RESULTS=$MASTER_RESULTS"
log "MASTER_RESULTS_DEBUG=$MASTER_RESULTS_DEBUG"
log "DETECT_FOREIGN_BUSY=$DETECT_FOREIGN_BUSY"
log "KILL_FILE=$KILL_FILE"
log "POLL_SEC=$POLL_SEC"
log "DRAIN_ON_EXIT=${DRAIN_ON_EXIT}"

for inst in "${FOLDERS[@]}"; do
  repo="${REPO_PREFIX}_${inst}"
  require_dir "$repo"
  require_file "$repo/tools/main.sh"
done

# Clear stale master pid tracking at startup
for inst in "${FOLDERS[@]}"; do
  pf="$(pid_file_for "$inst")"
  if [[ -f "$pf" ]]; then
    pid="$(cat "$pf" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && pid_alive "$pid"; then
      log "[inst=${inst}] Found master-tracked PID=${pid} (leaving it busy)"
    else
      log "[inst=${inst}] Clearing stale master PID tracking"
      clear_instance "$inst"
    fi
  fi
done

rebuild_done_from_master_results
clear_tasked_on_startup
rebuild_waiting_from_src_minus_done
update_kill_file

# Fill free slots initially
while true; do
  free_inst="$(find_free_instance)"
  [[ -n "$free_inst" ]] || break

  # SAFETY: ensure free_inst is a number
  if ! [[ "$free_inst" =~ ^[0-9]+$ ]]; then
    log "[queue] WARNING: free_inst is not numeric: '$free_inst' -> skipping scheduling cycle"
    sleep "$POLL_SEC"
    continue
  fi

  run_name="$(move_first_line_waiting_to_tasked)"
  [[ -n "$run_name" ]] || break

  set +e
  start_job_on_instance "$free_inst" "$run_name"
  rc=$?
  set -e

  if [[ "$rc" -eq 0 ]]; then
    continue
  fi

  log "[inst=${free_inst}] NOTE: start rc=${rc} for run=${run_name} -> removing from TASKED (will reappear in WAITING next start)"
  remove_from_tasked "$run_name"
  clear_instance "$free_inst"
  update_kill_file

  [[ "$rc" -eq 3 ]] && sleep "$POLL_SEC"
done

# Scheduler loop
while true; do
  any_running=false

  for inst in "${FOLDERS[@]}"; do
    pf="$(pid_file_for "$inst")"
    [[ -f "$pf" ]] || continue

    pid="$(cat "$pf" 2>/dev/null || true)"
    if [[ -z "$pid" ]]; then
      clear_instance "$inst"
      continue
    fi

    if pid_alive "$pid"; then
      any_running=true
      continue
    fi

    last_run="$(cat "$(run_file_for "$inst")" 2>/dev/null || true)"
    [[ -n "$last_run" ]] || { clear_instance "$inst"; continue; }

    set +e
    wait "$pid"
    rc=$?
    set -e
    echo "$rc" > "$(rc_file_for "$inst")" || true

    log "[inst=${inst}] DONE pid=${pid} run=${last_run} rc=${rc}"

    if move_results_to_central "$inst" "$last_run" "$rc"; then
      if [[ "$rc" -eq 0 ]]; then
        remove_from_tasked "$last_run"
        add_to_done "$last_run"
        log "[queue] TASKED->DONE for ${last_run} (success)"
      else
        log "[queue] FAILED rc=${rc} for ${last_run} (moved to MASTER_RESULTS_debug, not added to DONE)"
        remove_from_tasked "$last_run"
      fi
    else
      log "[queue] WARNING: results move failed for ${last_run} -> leaving it out of DONE"
      remove_from_tasked "$last_run"
    fi

    clear_instance "$inst"
    update_kill_file
  done

  # Schedule onto free instances
  while true; do
    free_inst="$(find_free_instance)"
    [[ -n "$free_inst" ]] || break

    if ! [[ "$free_inst" =~ ^[0-9]+$ ]]; then
      log "[queue] WARNING: free_inst is not numeric: '$free_inst' -> skipping scheduling cycle"
      break
    fi

    run_name="$(move_first_line_waiting_to_tasked)"
    [[ -n "$run_name" ]] || break

    set +e
    start_job_on_instance "$free_inst" "$run_name"
    rc=$?
    set -e

    if [[ "$rc" -eq 0 ]]; then
      any_running=true
      continue
    fi

    log "[inst=${free_inst}] NOTE: start rc=${rc} for run=${run_name} -> removing from TASKED (will reappear in WAITING next start)"
    remove_from_tasked "$run_name"
    clear_instance "$free_inst"
    update_kill_file

    [[ "$rc" -eq 3 ]] && break
  done

  next_waiting="$(norm_list "$WAITING_FILE" 2>/dev/null | head -n 1 || true)"
  if [[ "$any_running" == false && -z "$next_waiting" ]]; then
    log "All runs completed."
    log "==================== MASTER END ======================"
    update_kill_file
    exit 0
  fi

  update_kill_file
  sleep "$POLL_SEC"
done
