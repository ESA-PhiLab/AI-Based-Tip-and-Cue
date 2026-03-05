#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

# ===================== USER SETTINGS =====================

RUN_NAME="S_ramp"

# BASE_CONFIG="configs/deimv2/deimv2_dinov3_m_coco_whale.yml"
# PRETRAINED="ckpts/deimv2_dinov3_m_coco.pth"

BASE_CONFIG="configs/deimv2/deimv2_hgnetv2_atto_coco_whale.yml"
PRETRAINED="ckpts/deimv2_hgnetv2_atto_coco.pth"

TRAIN_ID="reflection_offnadir_glint_255"
TEST_ID="reflection_offnadir_glint_255"

DEFAULT_GPU_CHOICE="0"

TRAIN_CV="0"                 # 1|0
VAL_TEST_CV="0"              # 1|0 (rerun validation/test evaluation without training)
EVAL_AFTER_EACH_FOLD="1"     # 1|0
DUMP_COCO_JSON="0"           # 1|0
MAKE_OVERVIEW="1"            # 1|0

TRAIN_FINAL="0"              # 1|0
VAL_TEST_FINAL="0"           # 1|0 (rerun final validation/test evaluation without training)
FINAL_VAL_FRAC="0.05"
FINAL_SEED="42"
FINAL_MIN_VAL_PER_LOCATION="1"

CV_MODE="random"             # random|all
K_FOLDS="4"
SEED="42"
VAL_SIZE="2"

# Exactly 2 locations held out from CV and used as TEST
TEST_LOCATIONS="Ignacio2017,Auckland2006"

SELECT_METRIC="AP_precision_iou_0.50_area_all_maxdets_100"

EVAL_NAME="eval_data"
USE_AMP="1"                  # 1|0

USE_ENV_GPUS="${USE_ENV_GPUS:-1}"
FORCE_NPROC=""

OVERWRITE_OUTDIR="0"         # 1|0

# Count GPUs
NUM_GPUS=$(echo "$DEFAULT_GPU_CHOICE" | awk -F',' '{print NF}')


# Compute sum of GPU IDs
GPU_SUM=$(echo "$DEFAULT_GPU_CHOICE" | awk -F',' '{s=0; for(i=1;i<=NF;i++) s+=$i; print s}')

# Derive ports from sum (stable + low collision risk)
TRAIN_MASTER_PORT=$((10000 + GPU_SUM * 10 + 7))
EVAL_MASTER_PORT=$((10000 + GPU_SUM * 10 + 8))

# =================== DO NOT EDIT BELOW ===================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT_DIR="${REPO_ROOT}/results/${RUN_NAME}"
FINAL_OUT_DIR="${OUT_DIR}/final_location_holdout"

TRAIN_ROOT="${REPO_ROOT}/data/0_merged/${TRAIN_ID}"
TEST_ROOT="${REPO_ROOT}/data/0_merged/${TEST_ID}"

IMG_ROOT_TRAIN="${TRAIN_ROOT}"
IMG_ROOT_TEST="${TEST_ROOT}"

COCO_MERGED_TRAIN="${TRAIN_ROOT}/final_annotations_merged.json"
COCO_TEST_RAW="${TEST_ROOT}/final_annotations_merged.json"

# GPUs / nproc
if [[ "${USE_ENV_GPUS}" == "1" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  TRAIN_GPUS="${CUDA_VISIBLE_DEVICES}"
else
  TRAIN_GPUS="${DEFAULT_GPU_CHOICE}"
fi
EVAL_GPUS="${TRAIN_GPUS}"

TRAIN_NPROC="${FORCE_NPROC:-1}"
EVAL_NPROC="${FORCE_NPROC:-1}"

mkdir -p "${OUT_DIR}"

if [[ "${OVERWRITE_OUTDIR}" == "1" && -d "${OUT_DIR}" ]]; then
  echo "[main_local.sh] OVERWRITE_OUTDIR=1 -> deleting ${OUT_DIR}"
  rm -rf "${OUT_DIR}"
  mkdir -p "${OUT_DIR}"
fi

cp -f "${REPO_ROOT}/${BASE_CONFIG}" "${OUT_DIR}/" || true

# ---- Filter COCO_TEST to exactly the 2 TEST_LOCATIONS ----
COCO_TEST="${OUT_DIR}/test_holdout_only.json"

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

print(f"[main_local.sh] Filtered COCO_TEST -> {coco_out} | images={len(keep_images)} anns={len(keep_anns)} | keep={keep}")
PY

{
  echo "RUN_NAME=${RUN_NAME}"
  echo "BASE_CONFIG=${BASE_CONFIG}"
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

# ------------------ 1) Cross-validation training ------------------

CMD_CV=( python "${REPO_ROOT}/tools/train_crossval_deimv2.py"
  --img_root "${IMG_ROOT_TRAIN}"
  --img_root_test "${IMG_ROOT_TEST}"
  --coco_val "${COCO_MERGED_TRAIN}"
  --coco_test "${COCO_TEST}"
  --base_config "${BASE_CONFIG}"
  --pretrained "${PRETRAINED}"
  --results_dir "${OUT_DIR}"
  --test_locations "" \
  --holdout_test_locations "${TEST_LOCATIONS}" \
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

if [[ "${USE_AMP}" == "1" ]]; then
  CMD_CV+=( --use_amp )
fi

printf "%q " "${CMD_CV[@]}" > "${OUT_DIR}/cv_train.cmd.txt"
echo "" >> "${OUT_DIR}/cv_train.cmd.txt"

if [[ "${TRAIN_CV}" == "1" ]]; then
  echo "[main_local.sh] Cross-val training enabled (TRAIN_CV=1)"
  (cd "${REPO_ROOT}" && "${CMD_CV[@]}") | tee "${OUT_DIR}/cv_train_stdout.log"
else
  echo "[main_local.sh] Cross-val training skipped (TRAIN_CV=0)"
fi

# ------------------ 2) Final training ------------------

CMD_FINAL=( python "${REPO_ROOT}/tools/train_final.py"
  --img_root "${IMG_ROOT_TRAIN}"
  --repo_root "${REPO_ROOT}"
  --coco_trainval "${COCO_MERGED_TRAIN}"
  --base_config "${BASE_CONFIG}"
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
  echo "[main_local.sh] Final training enabled (TRAIN_FINAL=1)"
  (cd "${REPO_ROOT}" && "${CMD_FINAL[@]}") | tee "${FINAL_OUT_DIR}/train_stdout.log"
else
  echo "[main_local.sh] Final training skipped (TRAIN_FINAL=0)"
fi

# ------------------ 3) (Re)run validation/test evaluation (and dump predictions) ------------------

if [[ "${DUMP_COCO_JSON}" == "1" || "${VAL_TEST_CV}" == "1" ]]; then
  echo "[main_local.sh] (Re)running fold evaluation for VAL/TEST under cross_validation/ ..."
  python "${REPO_ROOT}/tools/dump_predictions_all.py" \
    --repo_root "${REPO_ROOT}" \
    --results_dir "${OUT_DIR}" \
    --base_config "${REPO_ROOT}/${BASE_CONFIG}" \
    --img_root "${IMG_ROOT_TRAIN}" \
    --eval_name "${EVAL_NAME}" \
    --split "both" \
    --gpus "${EVAL_GPUS}" \
    --nproc "${EVAL_NPROC}" \
    --master_port "${EVAL_MASTER_PORT}" \
    --overwrite "1" \
    --label_offset "0" \
    --include_final "0"
fi

if [[ "${DUMP_COCO_JSON}" == "1" || "${VAL_TEST_FINAL}" == "1" ]]; then
  echo "[main_local.sh] (Re)running final evaluation for VAL/TEST under final_location_holdout/ (if present) ..."
  python "${REPO_ROOT}/tools/dump_predictions_all.py" \
    --repo_root "${REPO_ROOT}" \
    --results_dir "${OUT_DIR}" \
    --base_config "${REPO_ROOT}/${BASE_CONFIG}" \
    --img_root "${IMG_ROOT_TRAIN}" \
    --eval_name "${EVAL_NAME}" \
    --split "both" \
    --gpus "${EVAL_GPUS}" \
    --nproc "${EVAL_NPROC}" \
    --master_port "$((EVAL_MASTER_PORT + 500))" \
    --overwrite "1" \
    --label_offset "0" \
    --include_final "1"
fi

# ------------------ 4) Overview from logs ------------------

if [[ "${MAKE_OVERVIEW}" == "1" ]]; then
  echo "[main_local.sh] Running evaluation/overview ..."
  python "${REPO_ROOT}/tools/evaluate_models.py" \
    --repo_root "${REPO_ROOT}" \
    --results_dir "${OUT_DIR}" \
    --eval_name "${EVAL_NAME}" \
    --select_metric "${SELECT_METRIC}" \
    --per_fold_only "0" \
    --overwrite "1" \
    --run_name "${RUN_NAME}"

else
  echo "[main_local.sh] MAKE_OVERVIEW=0 -> skipping overview"
fi

echo "[main_local.sh] DONE"