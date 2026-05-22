#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ZakharovNK/project"
STYLEGAN_ROOT="$PROJECT_ROOT/stylegan3-brecahad"
DATA_ROOT="$PROJECT_ROOT/gan_data_raw_zips"
OUT_ROOT="$PROJECT_ROOT/gan_training_runs_transfer"

PYTHON_BIN="python"
CFG="stylegan2"

# Сначала только самые перспективные классы
# vasc df
CLASSES=("vasc" "df")

GPUS=1
BATCH=16
GAMMA=0.8192
KIMG=2000
SNAP=10
MIRROR=1
METRICS="fid50k_full"

GLR=0.0025
DLR=0.0025
AUG="ada"
FREEZED=10
CBASE=16384

# Основной source net
RESUME_PKL="https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl"

mkdir -p "$OUT_ROOT"
cd "$STYLEGAN_ROOT"

echo "[INFO] TRANSFER LEARNING QUALITY RUN"
echo "[INFO] CFG=$CFG | BATCH=$BATCH | GAMMA=$GAMMA | KIMG=$KIMG | SNAP=$SNAP"
echo "[INFO] AUG=$AUG | FREEZED=$FREEZED | CBASE=$CBASE"
echo "[INFO] RESUME=$RESUME_PKL"

for cls in "${CLASSES[@]}"
do
    DATA_ZIP="$DATA_ROOT/${cls}_raw_256.zip"
    RUN_DIR="$OUT_ROOT/raw_${cls}_${CFG}_transfer_k${KIMG}_b${BATCH}"

    if [[ ! -f "$DATA_ZIP" ]]; then
        echo "[SKIP] $cls -> не найден датасет: $DATA_ZIP"
        continue
    fi

    if [[ -d "$RUN_DIR" ]]; then
        echo "[SKIP] $cls -> папка уже существует: $RUN_DIR"
        echo "       Удали её вручную перед повторным запуском"
        continue
    fi

    echo
    echo "============================================================"
    echo "[RUN] class=$cls"
    echo "[DATA] $DATA_ZIP"
    echo "[OUT ] $RUN_DIR"
    echo "============================================================"

    $PYTHON_BIN stylegan3/train.py \
        --outdir="$RUN_DIR" \
        --cfg="$CFG" \
        --data="$DATA_ZIP" \
        --gpus="$GPUS" \
        --batch="$BATCH" \
        --gamma="$GAMMA" \
        --mirror="$MIRROR" \
        --snap="$SNAP" \
        --kimg="$KIMG" \
        --metrics="$METRICS" \
        --glr="$GLR" \
        --dlr="$DLR" \
        --aug="$AUG" \
        --freezed="$FREEZED" \
        --cbase="$CBASE" \
        --resume="$RESUME_PKL"

    echo "[DONE] $cls"
done

echo
echo "[SUCCESS] Transfer quality runs finished"