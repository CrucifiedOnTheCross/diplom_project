#!/usr/bin/env bash
set -euo pipefail

# Runs the remaining GAN-related study steps without retraining completed experiments.
#
# It covers:
# 1) selected GAN-mix + focal weighted classifier runs;
# 2) confident-core GAN-mix dataset creation and two classifier runs;
# 3) calibrated real + GAN ensemble evaluation without training;
# 4) report refresh/package without forcing already existing reports.

SCIENCE_DIR="${SCIENCE_DIR:-science_folder}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"

REPORT_BATCH_SIZE="${REPORT_BATCH_SIZE:-128}"
FEATURE_BASELINE="${FEATURE_BASELINE:-baseline_real_only_raw_ganmix}"
OUTPUT_ARCHIVE="${OUTPUT_ARCHIVE:-experiment_results_extended_gan_study.tar.gz}"

EMB_ROOT="${EMB_ROOT:-embedding_analysis/24_raw_supcon_weighted}"
BASE_DATASET="${BASE_DATASET:-dataset}"
CONFIDENT_RATIO="${CONFIDENT_RATIO:-0.15}"
CONFIDENT_MIN_CONF="${CONFIDENT_MIN_CONF:-0.60}"
CONFIDENT_MAX_Q="${CONFIDENT_MAX_Q:-0.75}"
CONFIDENT_MIN_MARGIN="${CONFIDENT_MIN_MARGIN:-0.0}"

run_if_missing() {
  local exp_name="$1"
  local data_dir="$2"
  shift 2

  if [[ -f "$SCIENCE_DIR/$exp_name/best_model.pth" && -f "$SCIENCE_DIR/$exp_name/medical_metrics_report.txt" ]]; then
    echo "[SKIP] $exp_name already has best_model.pth and medical_metrics_report.txt"
    return
  fi

  rm -rf "$SCIENCE_DIR/$exp_name"
  echo "============================================================"
  echo "[TRAIN] $exp_name"
  echo "============================================================"
  python train.py \
    --out_dir "$SCIENCE_DIR" \
    --exp_name "$exp_name" \
    --data_dir "$data_dir" \
    --aug_type 1 \
    --batch_size "$BATCH_SIZE" \
    --accumulation_steps "$ACCUMULATION_STEPS" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --seed "$SEED" \
    "$@"
}

echo "============================================================"
echo "[1/5] GAN-mix focal weighted experiments"
echo "============================================================"

run_if_missing \
  gan_vasc_25_raw_ganmix_focal_weighted \
  datasets_gan_mix_raw/gan_vasc_25 \
  --loss focal --gamma 1.0 --use_weights

run_if_missing \
  gan_df_25_raw_ganmix_focal_weighted \
  datasets_gan_mix_raw/gan_df_25 \
  --loss focal --gamma 1.0 --use_weights

run_if_missing \
  gan_bcc50_akiec50_vasc25_raw_ganmix_focal_weighted \
  datasets_gan_mix_raw/gan_bcc50_akiec50_vasc25 \
  --loss focal --gamma 1.0 --use_weights

run_if_missing \
  random_ganmix_25_focal_weighted \
  datasets_feature_aware/random_ganmix_25 \
  --loss focal --gamma 1.0 --use_weights

echo
echo "============================================================"
echo "[2/5] Confident-core manifest and dataset"
echo "============================================================"

CONFIDENT_MANIFEST="synthetic_mixing_runs/confident_core_ganmix_15/manifest.csv"
CONFIDENT_DATASET="datasets_feature_aware/confident_core_ganmix_15"

if [[ ! -f "$CONFIDENT_MANIFEST" ]]; then
  python build_feature_aware_ganmix_manifest.py \
    --real-train-embeddings "$EMB_ROOT/train/embeddings.npy" \
    --real-train-metadata "$EMB_ROOT/train/metadata.csv" \
    --synthetic-embeddings "$EMB_ROOT/synthetic/embeddings.npy" \
    --synthetic-metadata "$EMB_ROOT/synthetic/metadata.csv" \
    --selection-mode confident_core \
    --synthetic-ratio "$CONFIDENT_RATIO" \
    --require-correct-pred \
    --min-confidence "$CONFIDENT_MIN_CONF" \
    --max-own-quantile "$CONFIDENT_MAX_Q" \
    --min-margin "$CONFIDENT_MIN_MARGIN" \
    --output "$CONFIDENT_MANIFEST"
else
  echo "[SKIP] Confident-core manifest exists: $CONFIDENT_MANIFEST"
fi

if [[ ! -d "$CONFIDENT_DATASET" ]]; then
  python build_manifest_dataset.py \
    --src-dataset "$BASE_DATASET" \
    --manifest "$CONFIDENT_MANIFEST" \
    --out-dataset "$CONFIDENT_DATASET" \
    --link-mode hardlink
else
  echo "[SKIP] Confident-core dataset exists: $CONFIDENT_DATASET"
fi

echo
echo "============================================================"
echo "[3/5] Confident-core classifier experiments"
echo "============================================================"

run_if_missing \
  confident_core_ganmix_15_raw_ce \
  "$CONFIDENT_DATASET" \
  --loss ce

run_if_missing \
  confident_core_ganmix_15_focal_weighted \
  "$CONFIDENT_DATASET" \
  --loss focal --gamma 1.0 --use_weights

echo
echo "============================================================"
echo "[4/5] Calibrated real + GAN ensemble"
echo "============================================================"

ENSEMBLE_NAME="${ENSEMBLE_NAME:-ensemble_real_supcon_ganmix}"
ENSEMBLE_MEMBERS="${ENSEMBLE_MEMBERS:-ablation_raw_focal_g1_weighted 24_raw_supcon_weighted gan_vasc_25_raw_ganmix random_ganmix_25_raw_ce}"

if [[ -f "$SCIENCE_DIR/$ENSEMBLE_NAME/medical_metrics_report_test.txt" ]]; then
  echo "[SKIP] Ensemble report exists: $SCIENCE_DIR/$ENSEMBLE_NAME/medical_metrics_report_test.txt"
else
  # shellcheck disable=SC2206
  ENSEMBLE_ARRAY=($ENSEMBLE_MEMBERS)
  python evaluate_ensemble.py \
    --science-dir "$SCIENCE_DIR" \
    --experiments "${ENSEMBLE_ARRAY[@]}" \
    --ensemble-name "$ENSEMBLE_NAME" \
    --test-dir dataset/test \
    --batch-size "$REPORT_BATCH_SIZE" \
    --device "$DEVICE"
fi

echo
echo "============================================================"
echo "[5/5] Reports and archive without recomputing existing reports"
echo "============================================================"

FEATURE_EXPERIMENTS="random_ganmix_25_raw_ce diverse_core_ganmix_25_raw_ce diverse_core_ganmix_25_weighted_ce random_ganmix_25_focal_weighted confident_core_ganmix_15_raw_ce confident_core_ganmix_15_focal_weighted" \
FEATURE_BASELINE="$FEATURE_BASELINE" \
BATCH_SIZE="$REPORT_BATCH_SIZE" \
FORCE_REPORTS=0 \
INCLUDE_DIPLOMA_FIGURES=0 \
OUTPUT_ARCHIVE="$OUTPUT_ARCHIVE" \
bash run_reports_and_package.sh

echo
echo "[SUCCESS] Extended GAN study finished."
