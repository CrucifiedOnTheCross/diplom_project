#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

SLEEP_SEC="${SLEEP_SEC:-300}"
PACKAGE_BUNDLE="${PACKAGE_BUNDLE:-0}"
PACKAGE_OUTPUT="${PACKAGE_OUTPUT:-latest_diploma_bundle.tar.gz}"

echo "[INFO] ROOT_DIR=$ROOT_DIR"
echo "[INFO] SLEEP_SEC=$SLEEP_SEC"
echo "[INFO] PACKAGE_BUNDLE=$PACKAGE_BUNDLE"

while true
do
    GPU_MEM="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)"
    GPU_UTIL="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1)"

    if [[ "$GPU_MEM" -lt 500 && "$GPU_UTIL" -lt 10 ]]; then
        DEVICE="cuda"
        echo "[INFO] GPU свободна (${GPU_MEM} MB, util=${GPU_UTIL}%), используем CUDA"
    else
        DEVICE="cpu"
        echo "[INFO] GPU занята (${GPU_MEM} MB, util=${GPU_UTIL}%), используем CPU"
    fi

    echo "[STEP] Calibration"
    python calibration_eval.py --device "$DEVICE"

    echo "[STEP] Threshold analysis (malignant)"
    python threshold_eval.py --mode malignant --criterion youden --device cpu

    echo "[STEP] Threshold analysis (melanoma)"
    python threshold_eval.py --mode melanoma --criterion youden --device cpu

    echo "[STEP] Collect summary"
    python update_experiment_summary.py

    echo "[STEP] Pareto / scatter plots"
    if [[ -f plot_pareto.py ]]; then
        python plot_pareto.py || true
    fi

    echo "[STEP] Threshold shift plots + tables"
    if [[ -f plot_threshold_shift.py ]]; then
        python plot_threshold_shift.py || true
    fi

    echo "[STEP] Clinical Pareto plots"
    if [[ -f plot_clinical_pareto.py ]]; then
        python plot_clinical_pareto.py || true
    fi

    if [[ "$PACKAGE_BUNDLE" == "1" ]]; then
        echo "[STEP] Package diploma bundle"
        if [[ -f package_diploma_results.sh ]]; then
            bash package_diploma_results.sh --output "$PACKAGE_OUTPUT" || true
        else
            echo "[WARN] package_diploma_results.sh not found, skipping package step"
        fi
    fi

    echo "[SLEEP] ${SLEEP_SEC} sec"
    sleep "$SLEEP_SEC"
done