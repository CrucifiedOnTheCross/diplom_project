#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="science_folder"
BASE_ROOT="datasets_gan_mix_raw"

EXPERIMENTS=(
  baseline_real_only
  gan_bcc_25
  gan_bcc_50
  gan_bcc_100
  gan_akiec_50
  gan_bcc50_akiec50
  gan_vasc_25
  gan_df_25
  gan_bcc100_akiec50
  gan_bcc50_akiec50_vasc25
)

for exp in "${EXPERIMENTS[@]}"
do
  echo "============================================================"
  echo "[RUN] $exp"
  echo "============================================================"

  python train.py \
    --out_dir "$OUT_DIR" \
    --exp_name "${exp}_raw_ganmix" \
    --data_dir "$BASE_ROOT/$exp" \
    --aug_type 1 \
    --batch_size 16 \
    --accumulation_steps 16 \
    --epochs 50 \
    --lr 1e-4 \
    --loss ce \
    --seed 42
done