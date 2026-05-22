#!/usr/bin/env bash
set -euo pipefail

# Run all post-experiment reports and package them into one downloadable archive.
#
# Typical usage:
#   bash run_reports_and_package.sh
#
# Optional overrides:
#   DEVICE=cpu bash run_reports_and_package.sh
#   FEATURE_BASELINE=01_base_raw bash run_reports_and_package.sh
#   OUTPUT_ARCHIVE=experiment_results_final.tar.gz bash run_reports_and_package.sh
#   INCLUDE_GRADCAM=1 INCLUDE_DIPLOMA_FIGURES=1 bash run_reports_and_package.sh

SCIENCE_DIR="${SCIENCE_DIR:-science_folder}"
DEVICE="${DEVICE:-cuda}"
THRESHOLD_CRITERION="${THRESHOLD_CRITERION:-youden}"
BATCH_SIZE="${BATCH_SIZE:-32}"

INCLUDE_FEATURE_AWARE="${INCLUDE_FEATURE_AWARE:-1}"
FEATURE_BASELINE="${FEATURE_BASELINE:-01_base_raw}"
FEATURE_EXPERIMENTS="${FEATURE_EXPERIMENTS:-random_ganmix_25_raw_ce diverse_core_ganmix_25_raw_ce diverse_core_ganmix_25_weighted_ce}"

INCLUDE_GRADCAM="${INCLUDE_GRADCAM:-0}"
INCLUDE_DIPLOMA_FIGURES="${INCLUDE_DIPLOMA_FIGURES:-0}"
INCLUDE_EMBEDDING_PLOTS="${INCLUDE_EMBEDDING_PLOTS:-1}"
INCLUDE_CHECKPOINTS="${INCLUDE_CHECKPOINTS:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ARCHIVE="${OUTPUT_ARCHIVE:-experiment_results_${STAMP}.tar.gz}"

echo "============================================================"
echo "[1/3] Post-experiment reports"
echo "============================================================"
echo "SCIENCE_DIR=$SCIENCE_DIR"
echo "DEVICE=$DEVICE"
echo "THRESHOLD_CRITERION=$THRESHOLD_CRITERION"
echo "BATCH_SIZE=$BATCH_SIZE"
echo

REPORT_CMD=(
  python make_all_reports.py
  --science-dir "$SCIENCE_DIR"
  --device "$DEVICE"
  --threshold-criterion "$THRESHOLD_CRITERION"
  --batch-size "$BATCH_SIZE"
)

if [[ "$INCLUDE_FEATURE_AWARE" == "1" ]]; then
  REPORT_CMD+=(
    --include-feature-aware
    --feature-aware-baseline "$FEATURE_BASELINE"
    --feature-aware-experiments
  )
  # shellcheck disable=SC2206
  FEATURE_EXPERIMENT_ARRAY=($FEATURE_EXPERIMENTS)
  REPORT_CMD+=("${FEATURE_EXPERIMENT_ARRAY[@]}")
fi

if [[ "$INCLUDE_GRADCAM" == "1" ]]; then
  REPORT_CMD+=(--include-gradcam)
fi

if [[ "$INCLUDE_DIPLOMA_FIGURES" == "1" ]]; then
  REPORT_CMD+=(--include-diploma-figures)
fi

printf '[RUN]'
printf ' %q' "${REPORT_CMD[@]}"
printf '\n'
"${REPORT_CMD[@]}"

echo
echo "============================================================"
echo "[2/3] Package archive"
echo "============================================================"
echo "OUTPUT_ARCHIVE=$OUTPUT_ARCHIVE"
echo

PACKAGE_CMD=(
  python package_experiment_archive.py
  --output "$OUTPUT_ARCHIVE"
  --science-dir "$SCIENCE_DIR"
)

if [[ "$INCLUDE_DIPLOMA_FIGURES" == "1" ]]; then
  PACKAGE_CMD+=(--include-diploma-figures)
fi

if [[ "$INCLUDE_EMBEDDING_PLOTS" == "1" ]]; then
  PACKAGE_CMD+=(--include-embedding-plots)
fi

if [[ "$INCLUDE_CHECKPOINTS" == "1" ]]; then
  PACKAGE_CMD+=(--include-checkpoints)
fi

printf '[RUN]'
printf ' %q' "${PACKAGE_CMD[@]}"
printf '\n'
"${PACKAGE_CMD[@]}"

echo
echo "============================================================"
echo "[3/3] Done"
echo "============================================================"
echo "Archive ready:"
echo "  $(pwd)/$OUTPUT_ARCHIVE"
