#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT_DIR="science_folder"

echo "================================================================="
echo " ЭТАП 6: СПАСЕНИЕ SUPCON (MEMORY-SAFE MODE)                      "
echo "================================================================="

# Физический батч снижаем до 16. С учетом n_views=2 в GPU полетит 32 картинки (это влезет).
# Шаги аккумуляции ставим 16. Итоговый эффективный батч: 16 * 16 = 256 (Как и было задумано!)

echo "--- Запуск: 12_prep_supcon ---"
python train.py --exp_name "12_prep_supcon" --loss "ce" --use_supcon --supcon_weight 0.1 --sampling_mode "none" --data_dir "dataset_preprocessed" --out_dir $OUT_DIR --batch_size 16 --accumulation_steps 16

echo "--- Запуск: 13_gan_supcon ---"
python train.py --exp_name "13_gan_supcon" --loss "ce" --use_supcon --supcon_weight 0.1 --sampling_mode "none" --data_dir "dataset_augmented" --out_dir $OUT_DIR --batch_size 16 --accumulation_steps 16

echo "================================================================="
echo " SUPCON УСПЕШНО ЗАВЕРШЕН!                                        "
echo "================================================================="