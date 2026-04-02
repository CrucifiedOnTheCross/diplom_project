#!/bin/bash
set -e

# УБРАЛИ "vasc", так как мы его доучим вручную командой выше
CLASSES=("df" "akiec" "bcc")

for CLS in "${CLASSES[@]}"; do
    echo "========================================================="
    echo " Начинаем обучение GAN для класса: $CLS (TURBO MODE)"
    echo "========================================================="
    
    python stylegan3-brecahad/stylegan3/dataset_tool.py \
        --source gan_data/${CLS}_raw \
        --dest gan_data/${CLS}_dataset_sg3.zip

    python stylegan3-brecahad/stylegan3/train.py \
        --outdir=gan_training_runs/${CLS}_sg3 \
        --cfg=stylegan2 \
        --data=gan_data/${CLS}_dataset_sg3.zip \
        --gpus=1 \
        --batch=32 \
        --batch-gpu=16 \
        --workers=16 \
        --mirror=1 \
        --snap=10 \
        --gamma=1.0 \
        --kimg=1000

    echo "Обучение для $CLS завершено успешно!"
done