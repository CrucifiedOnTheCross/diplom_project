#!/usr/bin/env bash
set -euo pipefail

# Упаковывает материалы диплома в один архив.
# По умолчанию НЕ включает тяжёлые checkpoint-файлы (*.pth).
#
# Примеры:
#   bash package_diploma_results.sh
#   bash package_diploma_results.sh --include-checkpoints
#   bash package_diploma_results.sh --output diploma_bundle_2026-04-15.tar.gz

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
SCIENCE_DIR="$ROOT_DIR/science_folder"
GAN_ANALYSIS_DIR="$ROOT_DIR/gan_analysis"
GAN_MIX_DIR="$ROOT_DIR/datasets_gan_mix_raw"
GAN_RUNS_DIR="$ROOT_DIR/gan_training_runs_transfer"
GRADCAM_DIR="$ROOT_DIR/gradcam_diploma"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_NAME="diploma_bundle_${STAMP}.tar.gz"
INCLUDE_CHECKPOINTS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --include-checkpoints)
      INCLUDE_CHECKPOINTS=1
      shift
      ;;
    --output)
      OUTPUT_NAME="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
  esac
done

TMP_DIR="$(mktemp -d)"
BUNDLE_DIR="$TMP_DIR/diploma_bundle"

mkdir -p "$BUNDLE_DIR/root_files"
mkdir -p "$BUNDLE_DIR/scripts"
mkdir -p "$BUNDLE_DIR/science_folder"
mkdir -p "$BUNDLE_DIR/gan_analysis"
mkdir -p "$BUNDLE_DIR/datasets_gan_mix_raw"
mkdir -p "$BUNDLE_DIR/gan_training_runs_transfer"
mkdir -p "$BUNDLE_DIR/gradcam_diploma"
mkdir -p "$BUNDLE_DIR/stylegan_metric_logs"

copy_if_exists () {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    echo "[ADD] $src"
  fi
}

# -----------------------------
# 1) Корневые summary / plots
# -----------------------------
ROOT_FILES=(
  "all_experiments_summary.csv"
  "threshold_clinical_summary.csv"

  "pareto_frontier_points.csv"
  "pareto_frontier_clean.png"
  "mcc_vs_ece_scatter_clean.png"
  "pareto_diploma.png"
  "pareto_legend.csv"

  "pareto_melanoma_sens_spec_after.png"
  "pareto_melanoma_sens_spec_after_legend.csv"
  "pareto_mcc_vs_melanoma_sens_after.png"
  "pareto_mcc_vs_melanoma_sens_after_legend.csv"

  "threshold_shift_vs_ece.png"
  "threshold_shift_vs_ece_malignant.png"
  "threshold_shift_vs_ece_melanoma.png"

  "calibration.log"
)

for f in "${ROOT_FILES[@]}"; do
  copy_if_exists "$ROOT_DIR/$f" "$BUNDLE_DIR/root_files/$f"
done

# -----------------------------
# 2) Ключевые скрипты проекта
# -----------------------------
SCRIPT_FILES=(
  "train.py"
  "model.py"
  "losses.py"
  "metrics.py"
  "calibration_eval.py"
  "threshold_eval.py"
  "update_experiment_summary.py"
  "plot_threshold_shift.py"
  "plot_pareto.py"
  "plot_clinical_pareto.py"
  "run_experiments.py"
  "analysis_loop.sh"
  "loop_metric.sh"
  "treshold_an.py"
  "eval_tta.py"

  "analyze_gan_fid_and_plan.py"
  "build_gan_augmented_datasets.py"
  "build_gradcam_diploma_tables.py"
  "gradcam_grid_by_class.py"
  "visualize_gradcam.py"

  "package_diploma_results.sh"
)

for f in "${SCRIPT_FILES[@]}"; do
  copy_if_exists "$ROOT_DIR/$f" "$BUNDLE_DIR/scripts/$f"
done

# -----------------------------
# 3) Сводки из science_folder
# -----------------------------
SCIENCE_TOP=(
  "calibration_summary.csv"
  "calibration_skip_log.csv"
  "threshold_summary_malignant_youden.csv"
  "threshold_summary_melanoma_youden.csv"
  "threshold_skip_log_malignant_youden.csv"
  "threshold_skip_log_melanoma_youden.csv"
)

for f in "${SCIENCE_TOP[@]}"; do
  copy_if_exists "$SCIENCE_DIR/$f" "$BUNDLE_DIR/science_folder/$f"
done

# -----------------------------
# 4) Отчёты по экспериментам классификатора
# -----------------------------
if [[ -d "$SCIENCE_DIR" ]]; then
  while IFS= read -r -d '' exp_dir; do
    exp_name="$(basename "$exp_dir")"
    out_dir="$BUNDLE_DIR/science_folder/$exp_name"
    mkdir -p "$out_dir"

    EXP_FILES=(
      "config.txt"
      "medical_metrics_report.txt"
      "calibration_report.json"
      "calibration_bins.csv"
      "threshold_report_malignant_youden.json"
      "threshold_report_melanoma_youden.json"
      "threshold_curve_before_malignant.csv"
      "threshold_curve_after_malignant.csv"
      "threshold_curve_before_melanoma.csv"
      "threshold_curve_after_melanoma.csv"
      "training_results.png"
    )

    for f in "${EXP_FILES[@]}"; do
      copy_if_exists "$exp_dir/$f" "$out_dir/$f"
    done

    if [[ "$INCLUDE_CHECKPOINTS" -eq 1 ]]; then
      copy_if_exists "$exp_dir/best_model.pth" "$out_dir/best_model.pth"
    fi
  done < <(find "$SCIENCE_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi

# -----------------------------
# 5) GAN analysis
# -----------------------------
GAN_ANALYSIS_FILES=(
  "gan_fid_history.csv"
  "gan_fid_summary.csv"
  "gan_best_snapshots.csv"
  "gan_single_class_plan.csv"
  "gan_combined_plan.csv"
  "gan_report.md"
  "generate_commands.sh"
)

for f in "${GAN_ANALYSIS_FILES[@]}"; do
  copy_if_exists "$GAN_ANALYSIS_DIR/$f" "$BUNDLE_DIR/gan_analysis/$f"
done

# -----------------------------
# 6) GAN-mix datasets summaries
# -----------------------------
copy_if_exists \
  "$GAN_MIX_DIR/_global_gan_dataset_summary.csv" \
  "$BUNDLE_DIR/datasets_gan_mix_raw/_global_gan_dataset_summary.csv"

if [[ -d "$GAN_MIX_DIR" ]]; then
  while IFS= read -r -d '' ds_dir; do
    ds_name="$(basename "$ds_dir")"
    out_dir="$BUNDLE_DIR/datasets_gan_mix_raw/$ds_name"
    mkdir -p "$out_dir"

    copy_if_exists "$ds_dir/_gan_summary.csv" "$out_dir/_gan_summary.csv"
    copy_if_exists "$ds_dir/_gan_manifest.csv" "$out_dir/_gan_manifest.csv"
  done < <(find "$GAN_MIX_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi

# -----------------------------
# 7) GAN training run summaries
# -----------------------------
if [[ -d "$GAN_RUNS_DIR" ]]; then
  while IFS= read -r -d '' outer_dir; do
    outer_name="$(basename "$outer_dir")"

    while IFS= read -r -d '' run_dir; do
      run_name="$(basename "$run_dir")"
      out_dir="$BUNDLE_DIR/gan_training_runs_transfer/$outer_name/$run_name"
      mkdir -p "$out_dir"

      GAN_RUN_FILES=(
        "metric-fid50k_full.jsonl"
        "training_options.json"
        "fakes_init.png"
        "reals.png"
        "stats.jsonl"
      )

      for f in "${GAN_RUN_FILES[@]}"; do
        copy_if_exists "$run_dir/$f" "$out_dir/$f"
      done
    done < <(find "$outer_dir" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

  done < <(find "$GAN_RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi

# -----------------------------
# 8) Grad-CAM diploma outputs
# -----------------------------
if [[ -d "$GRADCAM_DIR" ]]; then
  while IFS= read -r -d '' f; do
    rel="${f#$GRADCAM_DIR/}"
    copy_if_exists "$f" "$BUNDLE_DIR/gradcam_diploma/$rel"
  done < <(
    find "$GRADCAM_DIR" -type f \
      \( -name "*.csv" -o -name "*.png" -o -name "*.md" \) \
      -print0 | sort -z
  )
fi

# -----------------------------
# 9) StyleGAN metric logs
# -----------------------------
while IFS= read -r -d '' run_root; do
  while IFS= read -r -d '' f; do
    rel="${f#$ROOT_DIR/}"
    copy_if_exists "$f" "$BUNDLE_DIR/stylegan_metric_logs/$rel"
  done < <(
    find "$run_root" -type f \
      \( -name "metric-fid50k_full.jsonl" \
      -o -name "stats.jsonl" \
      -o -name "training_options.json" \
      -o -name "log.txt" \
      -o -name "fid*.jsonl" \) \
      -print0 | sort -z
  )
done < <(
  find "$ROOT_DIR" -maxdepth 1 -type d -name "gan_training_runs*" -print0 | sort -z
)

# -----------------------------
# 10) README для архива
# -----------------------------
cat > "$BUNDLE_DIR/README.txt" << EOF
Архив материалов диплома

Содержимое:
- root_files/: сводные CSV и графики
- scripts/: ключевые скрипты проекта
- science_folder/: отчёты и результаты по экспериментам классификатора
- gan_analysis/: анализ FID, планы mixing, команды генерации
- datasets_gan_mix_raw/: summary и manifest по собранным GAN-mix датасетам
- gan_training_runs_transfer/: основные метрики и служебные файлы transfer-GAN run'ов
- gradcam_diploma/: итоговые таблицы и визуализации Grad-CAM для диплома
- stylegan_metric_logs/: журналы FID и служебные логи обучения StyleGAN из всех папок gan_training_runs*

Примечание:
- checkpoint-файлы (*.pth) НЕ включены по умолчанию
- чтобы включить их, используй:
    bash package_diploma_results.sh --include-checkpoints
EOF

# -----------------------------
# 11) Упаковка
# -----------------------------
tar -czf "$ROOT_DIR/$OUTPUT_NAME" -C "$TMP_DIR" "diploma_bundle"

echo
echo "[SUCCESS] Archive created:"
echo "  $ROOT_DIR/$OUTPUT_NAME"

rm -rf "$TMP_DIR"