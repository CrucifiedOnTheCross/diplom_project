#!/usr/bin/env bash
set -euo pipefail

# Train the optional GAN-mix + SupCon experiment, then rebuild all downstream
# reports that are needed for the diploma analysis.
#
# Typical usage on the server:
#   bash run_supcon_ganmix_full_pipeline.sh
#
# Useful overrides:
#   EXP_NAME=diverse_core_ganmix_25_supcon_weighted DATA_DIR=datasets_feature_aware/diverse_core_ganmix_25 bash run_supcon_ganmix_full_pipeline.sh
#   EPOCHS=30 BATCH_SIZE=32 ACCUMULATION_STEPS=8 bash run_supcon_ganmix_full_pipeline.sh
#   FORCE_TRAIN=1 bash run_supcon_ganmix_full_pipeline.sh

SCIENCE_DIR="${SCIENCE_DIR:-science_folder}"
EXP_NAME="${EXP_NAME:-confident_core_ganmix_15_supcon_weighted}"
DATA_DIR="${DATA_DIR:-datasets_feature_aware/confident_core_ganmix_15}"
TEST_DIR="${TEST_DIR:-dataset/test}"

DEVICE="${DEVICE:-cuda}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"
AUG_TYPE="${AUG_TYPE:-1}"
SUPCON_WEIGHT="${SUPCON_WEIGHT:-0.1}"

REPORT_BATCH_SIZE="${REPORT_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SIGNIFICANCE_MODE="${SIGNIFICANCE_MODE:-fast}"
SIGNIFICANCE_WORKERS="${SIGNIFICANCE_WORKERS:-8}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
RUN_PACKAGE="${RUN_PACKAGE:-1}"
INCLUDE_EMBEDDING_PLOTS="${INCLUDE_EMBEDDING_PLOTS:-1}"
OUTPUT_ARCHIVE="${OUTPUT_ARCHIVE:-experiment_results_with_${EXP_NAME}.tar.gz}"

FEATURE_SPACE_EXPERIMENTS="${FEATURE_SPACE_EXPERIMENTS:-01_base_raw ablation_raw_focal_g1_weighted 24_raw_supcon_weighted gan_bcc50_akiec50_vasc25_raw_ganmix_focal_weighted ${EXP_NAME}}"

echo "============================================================"
echo "[0/8] Configuration"
echo "============================================================"
echo "SCIENCE_DIR=$SCIENCE_DIR"
echo "EXP_NAME=$EXP_NAME"
echo "DATA_DIR=$DATA_DIR"
echo "TEST_DIR=$TEST_DIR"
echo "DEVICE=$DEVICE"
echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
echo "ACCUMULATION_STEPS=$ACCUMULATION_STEPS"
echo "EPOCHS=$EPOCHS"
echo "LR=$LR"
echo "SEED=$SEED"
echo "AUG_TYPE=$AUG_TYPE"
echo "SUPCON_WEIGHT=$SUPCON_WEIGHT"
echo "REPORT_BATCH_SIZE=$REPORT_BATCH_SIZE"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "SIGNIFICANCE_MODE=$SIGNIFICANCE_MODE"
echo "SIGNIFICANCE_WORKERS=$SIGNIFICANCE_WORKERS"
echo "OUTPUT_ARCHIVE=$OUTPUT_ARCHIVE"
echo

run_cmd() {
  echo
  printf '[RUN]'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] Directory not found: $path" >&2
    exit 1
  fi
}

echo "============================================================"
echo "[1/8] Check dataset"
echo "============================================================"
require_dir "$DATA_DIR/train"
require_dir "$DATA_DIR/valid"
require_dir "$DATA_DIR/test"
require_dir "$TEST_DIR"

python - <<'PY' "$DATA_DIR"
from pathlib import Path
import sys

root = Path(sys.argv[1])
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
expected = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
for split in ["train", "valid", "test"]:
    split_dir = root / split
    missing = [cls for cls in expected if not (split_dir / cls).is_dir()]
    if missing:
        raise SystemExit(f"[ERROR] {split_dir} missing class dirs: {missing}")
    n = sum(1 for p in split_dir.rglob("*") if p.suffix.lower() in exts)
    print(f"[OK] {split_dir}: {n} images")
    if n == 0:
        raise SystemExit(f"[ERROR] {split_dir} has no images")

cache = root / "train" / "ham_cache_u8.pt"
if cache.exists():
    print(f"[INFO] Existing train cache: {cache}")
else:
    print("[INFO] No train cache yet; train.py will build it")
PY

echo
echo "============================================================"
echo "[2/8] Train experiment if needed"
echo "============================================================"
EXP_DIR="$SCIENCE_DIR/$EXP_NAME"
if [[ "$FORCE_TRAIN" != "1" && -s "$EXP_DIR/best_model.pth" && -s "$EXP_DIR/medical_metrics_report.txt" ]]; then
  echo "[SKIP] $EXP_NAME already has best_model.pth and medical_metrics_report.txt"
else
  run_cmd python train.py \
    --out_dir "$SCIENCE_DIR" \
    --exp_name "$EXP_NAME" \
    --data_dir "$DATA_DIR" \
    --aug_type "$AUG_TYPE" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --accumulation_steps "$ACCUMULATION_STEPS" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --loss ce \
    --use_weights \
    --use_supcon \
    --supcon_weight "$SUPCON_WEIGHT" \
    --seed "$SEED"
fi

echo
echo "============================================================"
echo "[3/8] Collect prediction cache for new experiment"
echo "============================================================"
run_cmd python collect_predictions.py \
  --science-dir "$SCIENCE_DIR" \
  --experiments "$EXP_NAME" \
  --device "$DEVICE" \
  --batch-size "$REPORT_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --only-missing

echo
echo "============================================================"
echo "[4/8] Rebuild medical, calibration, threshold, and summary reports"
echo "============================================================"
run_cmd python make_all_reports.py --stage medical --science-dir "$SCIENCE_DIR" --only-missing
run_cmd python make_all_reports.py --stage calibration --science-dir "$SCIENCE_DIR" --only-missing
run_cmd python make_all_reports.py --stage threshold --science-dir "$SCIENCE_DIR" --only-missing
run_cmd python make_all_reports.py --stage summary --science-dir "$SCIENCE_DIR"

echo
echo "============================================================"
echo "[5/8] Rebuild feature-aware comparison report"
echo "============================================================"
run_cmd python make_feature_aware_report.py \
  --summary all_experiments_summary.csv \
  --baseline 01_base_raw \
  --additional-baselines baseline_real_only_raw_ganmix \
  --experiments \
    random_ganmix_25_raw_ce \
    diverse_core_ganmix_25_raw_ce \
    diverse_core_ganmix_25_weighted_ce \
    confident_core_ganmix_15_raw_ce \
    confident_core_ganmix_15_focal_weighted \
    "$EXP_NAME" \
  --output-dir feature_aware_report

echo
echo "============================================================"
echo "[6/8] Run statistical significance"
echo "============================================================"
SIGNIFICANCE_ARGS=(--science-dir "$SCIENCE_DIR" --num-workers "$SIGNIFICANCE_WORKERS" --only-missing)
if [[ "$SIGNIFICANCE_MODE" == "final" ]]; then
  SIGNIFICANCE_ARGS+=(--final)
else
  SIGNIFICANCE_ARGS+=(--fast)
fi
run_cmd python run_significance_analysis.py "${SIGNIFICANCE_ARGS[@]}"

echo
echo "============================================================"
echo "[7/8] Run final feature-space comparison"
echo "============================================================"
# shellcheck disable=SC2206
FEATURE_SPACE_ARRAY=($FEATURE_SPACE_EXPERIMENTS)
run_cmd python make_feature_space_comparison.py \
  --science-dir "$SCIENCE_DIR" \
  --test-dir "$TEST_DIR" \
  --output-dir embedding_analysis/final_feature_space_comparison \
  --experiments "${FEATURE_SPACE_ARRAY[@]}" \
  --device "$DEVICE" \
  --batch-size "$REPORT_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS"

echo
echo "============================================================"
echo "[8/8] Package archive"
echo "============================================================"
if [[ "$RUN_PACKAGE" == "1" ]]; then
  if [[ -e "$OUTPUT_ARCHIVE" ]]; then
    BACKUP_ARCHIVE="${OUTPUT_ARCHIVE}.bak.$(date +%Y%m%d_%H%M%S)"
    echo "[WARN] Output archive already exists: $OUTPUT_ARCHIVE"
    echo "[WARN] Moving old archive to: $BACKUP_ARCHIVE"
    mv "$OUTPUT_ARCHIVE" "$BACKUP_ARCHIVE"
  fi

  PACKAGE_CMD=(python package_experiment_archive.py --output "$OUTPUT_ARCHIVE" --science-dir "$SCIENCE_DIR")
  if [[ "$INCLUDE_EMBEDDING_PLOTS" == "1" ]]; then
    PACKAGE_CMD+=(--include-embedding-plots)
  fi
  run_cmd "${PACKAGE_CMD[@]}"
  echo "[DONE] Archive: $(pwd)/$OUTPUT_ARCHIVE"
else
  echo "[SKIP] RUN_PACKAGE=$RUN_PACKAGE"
fi

echo
echo "============================================================"
echo "[DONE] Full SupCon GAN-mix pipeline finished"
echo "============================================================"
