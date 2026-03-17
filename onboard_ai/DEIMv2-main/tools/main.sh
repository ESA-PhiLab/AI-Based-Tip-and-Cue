#!/usr/bin/env bash
set -euo pipefail
trap '' HUP
LAUNCH_DIR="$(pwd)"

# ===================== USER SETTINGS =====================

FOLDER_ID=X
DEFAULT_GPU_CHOICE="X"

RUN_NAME="X"

BASE_CONFIG="configs/9_sizes/${RUN_NAME}.yml"
PRETRAINED="ckpts/deimv2_dinov3_s_coco.pth"

# BASE_CONFIG="configs/deimv2/deimv2_dinov3_m_coco_whale.yml"
# PRETRAINED="ckpts/deimv2_dinov3_m_coco.pth"

# BASE_CONFIG="configs/deimv2/deimv2_hgnetv2_femto_coco_whale.yml"
# PRETRAINED="ckpts/deimv2_hgnetv2_femto_coco.pth"

TRAIN_ID="reflection_offnadir_glint_255"
TEST_ID="reflection_offnadir_glint_255"

K_FOLDS="4"

TRAIN_CV="0"
TRAIN_FINAL="1"
VAL_TEST_CV="0"
VAL_TEST_FINAL="1"
DUMP_COCO_JSON="1"

EVAL_AFTER_EACH_FOLD="0"     # 1|0
MAKE_OVERVIEW="1"            # 1|0

COMPUTE_DATASET_STATS="1"   # 1|0
STATS_MAX_IMAGES="0"        # 0 => all images, >0 => debug limit

FINAL_VAL_FRAC="0.05"
FINAL_SEED="42"
FINAL_MIN_VAL_PER_LOCATION="1"

CV_MODE="random"             # random|all
SEED="42"
VAL_SIZE="2"

# ---- TensorBoard + plot export (optional) ----
EXPORT_TB_PLOTS="1"          # 1|0  export PNG+CSV from TB event files after training
START_TENSORBOARD="1"        # 1|0  start TB on remote (bind localhost) and print port-forward command
TB_PORT=""                  # empty => auto-derived from GPU_SUM (low collision)
TB_LOAD_FAST="0"            # 0 => pass --load_fast=false (more stable on NFS); 1 => default
TB_LOGDIR_MODE="all"        # all|cv|fold1|final  (what TB should show)
# ---------------------------------------------

# Image-level threshold used for "has any box" metrics (precision/recall/F1).
# If OPTIMIZE_SCORE_THR=1, the best threshold is chosen per run (max image-level F1) and this is ignored.
SCORE_THR="0.05"
OPTIMIZE_SCORE_THR="1"   # 1 => optimize for image-level F1, 0 => use SCORE_THR

# Exactly 2 locations that are held out from CV and used as TEST set.
# Allowed: empty (no test) OR exactly 2 locations.
TEST_LOCATIONS="Pelagos2016,Auckland2006"

SELECT_METRIC="AP_precision_iou_0.50_area_all_maxdets_100"

EVAL_NAME="eval_data"
EVAL_SPLIT="${EVAL_SPLIT:-both}"   # val|test|both

USE_AMP="1"                  # 1|0

USE_ENV_GPUS="${USE_ENV_GPUS:-1}"
FORCE_NPROC=""

OVERWRITE_OUTDIR="0"         # 1|0

# =================== DO NOT EDIT BELOW ===================



ALL_LOCATIONS="Auckland2006,Auckland2011,Ignacio2017,Maui2015,Pelagos2016,Valdes2012,Valdes2014,Valdes2016,Witsand2009"
if [[ -z "${TEST_LOCATIONS// }" ]]; then
  TRAIN_LOCATIONS="${ALL_LOCATIONS}"
else
  TRAIN_LOCATIONS="$(echo "$ALL_LOCATIONS" | tr ',' '\n' | grep -v -E "$(echo "$TEST_LOCATIONS" | tr ',' '|')" | paste -sd "," -)"
fi



SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT_DIR="${REPO_ROOT}/results/${RUN_NAME}"
FINAL_OUT_DIR="${OUT_DIR}/final_location_holdout"

PORT_FILE="${LAUNCH_DIR}/port_forward.txt"

TRAIN_ROOT="${REPO_ROOT}/data/0_merged/${TRAIN_ID}"
TEST_ROOT="${REPO_ROOT}/data/0_merged/${TEST_ID}"

IMG_ROOT_TRAIN="${TRAIN_ROOT}"
IMG_ROOT_TEST="${TEST_ROOT}"

COCO_MERGED_TRAIN="${TRAIN_ROOT}/final_annotations_repaired.json"
COCO_TEST_RAW="${TEST_ROOT}/final_annotations_repaired.json"

# ---------------- GPUs (set mask early) + safer ports ----------------

# Decide final physical GPU mask early so ports/TB cleanup are correct.
if [[ "${USE_ENV_GPUS}" == "1" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_PHYS="${CUDA_VISIBLE_DEVICES}"     # honor launch env (physical ids)
else
  CUDA_PHYS="${DEFAULT_GPU_CHOICE}"       # use script setting (physical ids)
fi
export CUDA_VISIBLE_DEVICES="${CUDA_PHYS}"

_port_hash() { echo -n "$1" | cksum | awk '{print $1 % 1000}'; }

_find_free_port() {
  local p="$1"
  while true; do
    if command -v lsof >/dev/null 2>&1; then
      lsof -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1 || { echo "$p"; return; }
    elif command -v ss >/dev/null 2>&1; then
      ss -ltn "( sport = :$p )" 2>/dev/null | tail -n +2 | grep -q . || { echo "$p"; return; }
    else
      echo "$p"; return
    fi
    p=$((p + 1))
    [[ "$p" -lt 65535 ]] || { echo "No free port found" 1>&2; return 1; }
  done
}

PORT_KEY="${FOLDER_ID}|${CUDA_PHYS}|${RUN_NAME}"
H=$(_port_hash "${PORT_KEY}")

TRAIN_MASTER_PORT=$(_find_free_port $(( 15000 + FOLDER_ID * 1000 + H )))
EVAL_MASTER_PORT=$(_find_free_port  $(( 25000 + FOLDER_ID * 1000 + H )))
if [[ -z "${TB_PORT}" ]]; then
  TB_PORT=$(_find_free_port       $(( 35000 + FOLDER_ID * 1000 + H )))
else
  TB_PORT=$(_find_free_port "${TB_PORT}")
fi

echo "[ports] train=${TRAIN_MASTER_PORT} eval=${EVAL_MASTER_PORT} tb=${TB_PORT} key=${PORT_KEY} hash=${H}"
# ---------------------------------------------------------------------



# --------- TensorBoard safety net (cleanup from previous runs) ---------
cleanup_old_tensorboard() {
  mkdir -p "${OUT_DIR}"

  # 1) Kill previous TB by PID file (if still alive)
  if [[ -f "${OUT_DIR}/tensorboard.pid" ]]; then
    local old_pid
    old_pid="$(cat "${OUT_DIR}/tensorboard.pid" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" >/dev/null 2>&1; then
      echo "[main.sh] Found old TensorBoard PID ${old_pid} -> stopping it"
      kill "${old_pid}" >/dev/null 2>&1 || true
      sleep 1
      kill -0 "${old_pid}" >/dev/null 2>&1 && kill -9 "${old_pid}" >/dev/null 2>&1 || true
    fi
    rm -f "${OUT_DIR}/tensorboard.pid" >/dev/null 2>&1 || true
  fi

  # 2) Kill anything currently bound to TB_PORT (PID file might be stale)
  if command -v lsof >/dev/null 2>&1; then
    local pids
    pids="$(lsof -t -iTCP:"${TB_PORT}" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
      echo "[main.sh] Port ${TB_PORT} already in use by PID(s): ${pids} -> stopping them"
      # term first, then hard-kill if needed
      kill ${pids} >/dev/null 2>&1 || true
      sleep 1
      kill -9 ${pids} >/dev/null 2>&1 || true
    fi
  else
    echo "[main.sh] NOTE: lsof not found; cannot auto-kill processes bound to TB_PORT=${TB_PORT}"
    echo "[main.sh] Install lsof or rely on tensorboard.pid cleanup only."
  fi
}

cleanup_old_tensorboard
# ---------------------------------------------------------------------



# --------- TensorBoard cleanup (auto-stop on exit) ---------
schedule_tensorboard_shutdown() {

  if [[ ! -f "${OUT_DIR}/tensorboard.pid" ]]; then
    return
  fi

  tb_pid="$(cat "${OUT_DIR}/tensorboard.pid" 2>/dev/null || true)"

  if [[ -z "${tb_pid}" ]]; then
    return
  fi

  echo "[main.sh] TensorBoard will be stopped automatically in 48 hours (PID ${tb_pid})"

  (
    sleep $((48 * 60 * 60))

    if kill -0 "${tb_pid}" >/dev/null 2>&1; then
      echo "[main.sh] Auto-stopping TensorBoard after 48 hours (PID ${tb_pid})"
      kill "${tb_pid}" >/dev/null 2>&1 || true
    fi

    rm -f "${OUT_DIR}/tensorboard.pid" >/dev/null 2>&1 || true

  ) &
}

trap 'schedule_tensorboard_shutdown' EXIT INT TERM
# ----------------------------------------------------------
# ----------------------------------------------------------

# If no TEST locations are configured, force eval split away from test.
if [[ -z "${TEST_LOCATIONS// }" ]]; then
  if [[ "${EVAL_SPLIT}" == "test" || "${EVAL_SPLIT}" == "both" ]]; then
    echo "[main.sh] NOTE: TEST_LOCATIONS is empty -> forcing EVAL_SPLIT=val" 1>&2
    EVAL_SPLIT="val"
  fi
fi


# TENSORBOARD
tb_logdir_for_mode() {
  local mode="$1"
  case "$mode" in
    all)   echo "${OUT_DIR}" ;;
    cv)    echo "${OUT_DIR}/cross_validation" ;;
    fold1) echo "${OUT_DIR}/cross_validation/fold1/summary" ;;
    final) echo "${FINAL_OUT_DIR}/summary" ;;
    *)     echo "${OUT_DIR}/cross_validation" ;;
  esac
}

# GPUs / nproc  (PASS PHYSICAL IDs to python launchers; they set CUDA_VISIBLE_DEVICES themselves)

# Decide final physical GPU mask (honor launch env if requested)
if [[ "${USE_ENV_GPUS}" == "1" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_PHYS="${CUDA_VISIBLE_DEVICES}"     # physical ids from launch
else
  CUDA_PHYS="${DEFAULT_GPU_CHOICE}"       # physical ids from script
fi

# IMPORTANT: pass PHYSICAL ids to the python tools, because they set CUDA_VISIBLE_DEVICES = --gpus
TRAIN_GPUS="${CUDA_PHYS}"
EVAL_GPUS="${CUDA_PHYS}"

# nproc = number of visible GPUs in the mask (unless forced)
NUM_VIS=$(echo "${CUDA_PHYS}" | awk -F',' '{print NF}')
TRAIN_NPROC="${FORCE_NPROC:-${NUM_VIS}}"
EVAL_NPROC="${FORCE_NPROC:-${NUM_VIS}}"


mkdir -p "${OUT_DIR}"

if [[ "${OVERWRITE_OUTDIR}" == "1" && -d "${OUT_DIR}" ]]; then
  echo "[main.sh] OVERWRITE_OUTDIR=1 -> deleting ${OUT_DIR}"
  rm -rf "${OUT_DIR}"
  mkdir -p "${OUT_DIR}"
fi

# ---- Freeze the run config inside results/ and ONLY use that from now on ----

ORIG_BASE_CONFIG="${BASE_CONFIG}"
RUN_CONFIG="${OUT_DIR}/${RUN_NAME}.yml"
cp -f "${REPO_ROOT}/${ORIG_BASE_CONFIG}" "${RUN_CONFIG}"

# Rewrite __include__ entries to ABSOLUTE paths so moving the config never breaks includes
export REPO_ROOT_ABS="${REPO_ROOT}"
export ORIG_CFG_ABS="${REPO_ROOT}/${ORIG_BASE_CONFIG}"
export RUN_CFG_ABS="${RUN_CONFIG}"

python - <<'PY'
import os, re
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT_ABS"]).resolve()
orig_cfg = Path(os.environ["ORIG_CFG_ABS"]).resolve()
run_cfg = Path(os.environ["RUN_CFG_ABS"]).resolve()
orig_dir = orig_cfg.parent

txt = run_cfg.read_text(encoding="utf-8")

# Find __include__ flow-list block: __include__: [ ... ]
m = re.search(r"(?s)__include__\s*:\s*\[(.*?)\]", txt)
if not m:
    raise SystemExit(f"[main.sh] ERROR: could not find __include__: [...] in {run_cfg}")

inner = m.group(1)

# Replace each quoted path inside include list
def repl(match: re.Match) -> str:
    q = match.group(1)
    p = match.group(2)
    pth = Path(p)

    # keep absolute paths as-is
    if pth.is_absolute():
        return f'{q}{str(pth)}{q}'

    # resolve relative to the ORIGINAL config directory
    abs_p = (orig_dir / pth).resolve()
    if not abs_p.exists():
        raise SystemExit(f"[main.sh] ERROR: include does not exist after resolve: {p} -> {abs_p}")

    return f'{q}{str(abs_p)}{q}'

inner2 = re.sub(r"(['\"])([^'\"]+)\1", repl, inner)

txt2 = txt[:m.start(1)] + inner2 + txt[m.end(1):]
run_cfg.write_text(txt2, encoding="utf-8")

print(f"[main.sh] Frozen config with absolute includes -> {run_cfg}")
PY

echo "[main.sh] Using run config: ${RUN_CONFIG}"
BASE_CONFIG="${RUN_CONFIG}"


# ---- Optional: Filter COCO_TEST to exactly the TEST_LOCATIONS ----
COCO_TEST="${OUT_DIR}/test_holdout_only.json"

if [[ -z "${TEST_LOCATIONS// /}" ]]; then
  echo "[main.sh] TEST_LOCATIONS is empty -> no TEST split will be used."
  COCO_TEST=""
else
  export COCO_TEST_IN="${COCO_TEST_RAW}"
  export COCO_TEST_OUT="${COCO_TEST}"
  export TEST_LOCS="${TEST_LOCATIONS}"

  python - <<'PY'
import json, os
from pathlib import Path

def _match(fn: str, loc: str) -> bool:
    fn = (fn or "").replace("\\", "/").lstrip("./").lstrip("/")
    parts = set(Path(fn).parts)
    if loc in parts:
        return True
    if f"/{loc}/" in fn:
        return True
    if fn.startswith(loc + "/"):
        return True
    return False

coco_in = Path(os.environ["COCO_TEST_IN"])
coco_out = Path(os.environ["COCO_TEST_OUT"])
keep = [x.strip() for x in os.environ["TEST_LOCS"].split(",") if x.strip()]

if len(keep) != 2:
    raise SystemExit(f"TEST_LOCATIONS must contain exactly 2 items, got {len(keep)}: {keep}")

data = json.loads(coco_in.read_text(encoding="utf-8"))
images = data.get("images", []) or []
anns = data.get("annotations", []) or []

keep_ids = set()
keep_images = []
for im in images:
    fn = im.get("file_name", "")
    if any(_match(fn, loc) for loc in keep):
        iid = int(im["id"])
        keep_ids.add(iid)
        keep_images.append(im)

keep_anns = [a for a in anns if int(a.get("image_id")) in keep_ids]

out = {}
for k in ["info", "licenses", "categories"]:
    if k in data:
        out[k] = data[k]
out["images"] = keep_images
out["annotations"] = keep_anns

coco_out.parent.mkdir(parents=True, exist_ok=True)
coco_out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"[main.sh] Filtered COCO_TEST -> {coco_out} | images={len(keep_images)} anns={len(keep_anns)} | keep={keep}")
PY
fi

# capture config
{
  echo "RUN_NAME=${RUN_NAME}"
  echo "BASE_CONFIG=${BASE_CONFIG}  # run-local frozen config"
  echo "PRETRAINED=${PRETRAINED}"
  echo "TRAIN_ID=${TRAIN_ID}"
  echo "TEST_ID=${TEST_ID}"
  echo "TRAIN_CV=${TRAIN_CV}"
  echo "VAL_TEST_CV=${VAL_TEST_CV}"
  echo "EVAL_AFTER_EACH_FOLD=${EVAL_AFTER_EACH_FOLD}"
  echo "DUMP_COCO_JSON=${DUMP_COCO_JSON}"
  echo "MAKE_OVERVIEW=${MAKE_OVERVIEW}"
  echo "TRAIN_FINAL=${TRAIN_FINAL}"
  echo "VAL_TEST_FINAL=${VAL_TEST_FINAL}"
  echo "K_FOLDS=${K_FOLDS}"
  echo "SEED=${SEED}"
  echo "VAL_SIZE=${VAL_SIZE}"
  echo "TEST_LOCATIONS=${TEST_LOCATIONS}"
  echo "SELECT_METRIC=${SELECT_METRIC}"
  echo "TRAIN_GPUS=${TRAIN_GPUS}"
  echo "EVAL_GPUS=${EVAL_GPUS}"
  echo "TRAIN_NPROC=${TRAIN_NPROC}"
  echo "EVAL_NPROC=${EVAL_NPROC}"
  echo "TRAIN_MASTER_PORT=${TRAIN_MASTER_PORT}"
  echo "EVAL_MASTER_PORT=${EVAL_MASTER_PORT}"
  echo "EVAL_NAME=${EVAL_NAME}"
  echo "USE_AMP=${USE_AMP}"
  echo "COCO_TEST_RAW=${COCO_TEST_RAW}"
  echo "COCO_TEST_FILTERED=${COCO_TEST}"
} > "${OUT_DIR}/launcher_config.txt"


if [[ "${COMPUTE_DATASET_STATS}" == "1" ]]; then
  echo "[main.sh] Computing dataset RGB mean/std from COCO_MERGED_TRAIN ..."
  python "${REPO_ROOT}/tools/compute_dataset_mean_std.py" \
  --img_root "${IMG_ROOT_TRAIN}" \
  --coco "${COCO_MERGED_TRAIN}" \
  --locations "${TRAIN_LOCATIONS}" \
  --out_json "${OUT_DIR}/dataset_rgb_mean_std.json" \
  | tee "${OUT_DIR}/dataset_rgb_mean_std.log"
fi


# ------------------ TensorBoard (start early) + port_forward.txt ------------------



write_port_file() {
  local logdir="$1"
  {
    echo "TensorBoard monitoring"
    echo "======================"
    echo ""
    echo "Run this on your LOCAL machine:"
    echo ""
    echo "ssh -L ${TB_PORT}:127.0.0.1:${TB_PORT} iv-mind"
    echo ""
    echo "Then open in your browser:"
    echo ""
    echo "http://localhost:${TB_PORT}"
    echo ""
    echo "TensorBoard logdir:"
    echo "${logdir}"
    echo ""
    echo "Run directory:"
    echo "${OUT_DIR}"
  } > "${PORT_FILE}"
}

start_tensorboard_early() {
  local logdir="$1"

  echo "[main.sh] Starting TensorBoard early (REMOTE localhost only) ..."
  echo "[main.sh]   logdir: ${logdir}"
  echo "[main.sh]   port:   ${TB_PORT}"

  # Kill previous TB started by this run (best-effort)
  if [[ -f "${OUT_DIR}/tensorboard.pid" ]]; then
    old_pid="$(cat "${OUT_DIR}/tensorboard.pid" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]]; then
      kill "${old_pid}" >/dev/null 2>&1 || true
    fi
    rm -f "${OUT_DIR}/tensorboard.pid" || true
  fi

  TB_FAST_FLAG=()
  if [[ "${TB_LOAD_FAST}" == "0" ]]; then
    TB_FAST_FLAG+=( --load_fast=false )
  fi

  nohup python -m tensorboard.main \
    --logdir "${logdir}" \
    --port "${TB_PORT}" \
    --host 127.0.0.1 \
    "${TB_FAST_FLAG[@]}" \
    > "${OUT_DIR}/tensorboard_stdout.log" 2>&1 &

  echo $! > "${OUT_DIR}/tensorboard.pid"

  write_port_file "${logdir}"

  echo "[main.sh] TensorBoard started (PID $(cat "${OUT_DIR}/tensorboard.pid"))"
  echo "[main.sh] Port forwarding instructions written to: ${PORT_FILE}"
}

# Start TB BEFORE training so you can live-follow
if [[ "${START_TENSORBOARD}" == "1" ]]; then
  TB_LOGDIR="$(tb_logdir_for_mode "${TB_LOGDIR_MODE}")"
  start_tensorboard_early "${TB_LOGDIR}"
else
  # still write instructions file with expected logdir/port (useful even if TB is off)
  TB_LOGDIR="$(tb_logdir_for_mode "${TB_LOGDIR_MODE}")"
  write_port_file "${TB_LOGDIR}"
  echo "[main.sh] START_TENSORBOARD=0 -> wrote ${PORT_FILE} anyway"
fi

# -------------------------------------------------------------------------------


# ------------------ 1) Cross-validation training ------------------

CMD_CV=( python "${REPO_ROOT}/tools/train_crossval_deimv2.py"
  --img_root "${IMG_ROOT_TRAIN}"
  --img_root_test "${IMG_ROOT_TEST}"
  --coco_val "${COCO_MERGED_TRAIN}"
  --coco_test "${COCO_TEST}"
  --base_config "${RUN_CONFIG}"
  --pretrained "${PRETRAINED}"
  --results_dir "${OUT_DIR}"
  --test_locations ""
  --holdout_test_locations "${TEST_LOCATIONS}"
  --mode "${CV_MODE}"
  --k "${K_FOLDS}"
  --seed "${SEED}"
  --val_size "${VAL_SIZE}"
  --gpus "${TRAIN_GPUS}"
  --nproc "${TRAIN_NPROC}"
  --master_port "${TRAIN_MASTER_PORT}"
  --eval_after_each_fold "${EVAL_AFTER_EACH_FOLD}"
  --eval_name "${EVAL_NAME}"
  --eval_gpus "${EVAL_GPUS}"
  --eval_nproc "${EVAL_NPROC}"
  --eval_master_port "${EVAL_MASTER_PORT}"
  --overwrite_eval "1"
  --select_metric "${SELECT_METRIC}"
  --label_offset "0"
)


# If no test locations were provided, remove test args entirely (train+val only)
if [[ -z "${TEST_LOCATIONS// /}" ]]; then
  # remove --coco_test <val> and --holdout_test_locations <val>
  for i in "${!CMD_CV[@]}"; do
    v="${CMD_CV[$i]-}"
    if [[ "${v}" == "--coco_test" || "${v}" == "--holdout_test_locations" ]]; then
      unset "CMD_CV[$i]"
      j=$((i+1))
      if [[ -n "${CMD_CV[$j]-}" ]]; then
        unset "CMD_CV[$j]"
      fi
    fi
  done
  # compact array
  CMD_CV=("${CMD_CV[@]}")
fi

if [[ "${USE_AMP}" == "1" ]]; then
  CMD_CV+=( --use_amp )
fi

printf "%q " "${CMD_CV[@]}" > "${OUT_DIR}/cv_train.cmd.txt"
echo "" >> "${OUT_DIR}/cv_train.cmd.txt"

if [[ "${TRAIN_CV}" == "1" ]]; then
  echo "[main.sh] Cross-val training enabled (TRAIN_CV=1)"
  (cd "${REPO_ROOT}" && "${CMD_CV[@]}") | tee "${OUT_DIR}/cv_train_stdout.log"
else
  echo "[main.sh] Cross-val training skipped (TRAIN_CV=0)"
fi

# ------------------ 2) Final training ------------------

CMD_FINAL=( python "${REPO_ROOT}/tools/train_final.py"
  --img_root "${IMG_ROOT_TRAIN}"
  --repo_root "${REPO_ROOT}"
  --coco_trainval "${COCO_MERGED_TRAIN}"
  --base_config "${RUN_CONFIG}"
  --pretrained "${PRETRAINED}"
  --output_dir "${FINAL_OUT_DIR}"
  --val_frac "${FINAL_VAL_FRAC}"
  --seed "${FINAL_SEED}"
  --min_val_per_location "${FINAL_MIN_VAL_PER_LOCATION}"
  --gpus "${TRAIN_GPUS}"
  --nproc "${TRAIN_NPROC}"
  --master_port "$((TRAIN_MASTER_PORT + 200))"
  --eval_name "${EVAL_NAME}"
  --eval_gpus "${EVAL_GPUS}"
  --eval_nproc "${EVAL_NPROC}"
  --eval_master_port "$((EVAL_MASTER_PORT + 200))"
  --overwrite "0"
  --overwrite_eval "1"
  --coco_test "${COCO_TEST}"
  --img_root_test "${IMG_ROOT_TEST}"
  --val_test_final "${VAL_TEST_FINAL}"
  --label_offset "0"
)

if [[ "${USE_AMP}" == "1" ]]; then
  CMD_FINAL+=( --use_amp )
fi

printf "%q " "${CMD_FINAL[@]}" > "${OUT_DIR}/final_train.cmd.txt"
echo "" >> "${OUT_DIR}/final_train.cmd.txt"

if [[ "${TRAIN_FINAL}" == "1" ]]; then
  echo "[main.sh] Final training enabled (TRAIN_FINAL=1)"
  (cd "${REPO_ROOT}" && "${CMD_FINAL[@]}") | tee "${OUT_DIR}/final_train_stdout.log"
else
  echo "[main.sh] Final training skipped (TRAIN_FINAL=0)"
fi

# ------------------ 3) (Re)run validation/test evaluation (and dump predictions) ------------------

if [[ "${VAL_TEST_CV}" == "1" ]]; then
  echo "[main.sh] (Re)running fold evaluation for VAL/TEST under cross_validation/ ..."
  python "${REPO_ROOT}/tools/dump_predictions_all.py" \
  --repo_root "${REPO_ROOT}" \
  --results_dir "${OUT_DIR}" \
  --img_root "${IMG_ROOT_TRAIN}" \
  --img_root_test "${IMG_ROOT_TEST}" \
  --eval_name "${EVAL_NAME}" \
  --split "${EVAL_SPLIT}" \
  --gpus "${EVAL_GPUS}" \
  --nproc "${EVAL_NPROC}" \
  --master_port "${EVAL_MASTER_PORT}" \
  --overwrite "1" \
  --label_offset "0" \
  --include_final "0" \
  --score_thr "${SCORE_THR}" \
  --optimize_score_thr "${OPTIMIZE_SCORE_THR}"
fi

if [[ "${VAL_TEST_FINAL}" == "1" ]]; then
  echo "[main.sh] (Re)running final evaluation for VAL/TEST under final_location_holdout/ ..."
  python "${REPO_ROOT}/tools/dump_predictions_all.py" \
    --repo_root "${REPO_ROOT}" \
    --results_dir "${OUT_DIR}" \
    --img_root "${IMG_ROOT_TRAIN}" \
    --img_root_test "${IMG_ROOT_TEST}" \
    --eval_name "${EVAL_NAME}" \
    --split "${EVAL_SPLIT}" \
    --gpus "${EVAL_GPUS}" \
    --nproc "${EVAL_NPROC}" \
    --master_port "$((EVAL_MASTER_PORT + 500))" \
    --overwrite "1" \
    --label_offset "0" \
    --include_final "1" \
    --score_thr "${SCORE_THR}" \
    --optimize_score_thr "${OPTIMIZE_SCORE_THR}"
fi

# ------------------ 4) Overview from logs ------------------

if [[ "${MAKE_OVERVIEW}" == "1" ]]; then
  echo "[main.sh] Running evaluation/overview ..."
  python "${REPO_ROOT}/tools/evaluate_models.py" \
    --repo_root "${REPO_ROOT}" \
    --results_dir "${OUT_DIR}" \
    --eval_name "${EVAL_NAME}" \
    --select_metric "${SELECT_METRIC}" \
    --per_fold_only "0" \
    --overwrite "1" \
    --run_name "${RUN_NAME}"
else
  echo "[main.sh] MAKE_OVERVIEW=0 -> skipping overview"
fi

echo "[main.sh] DONE"