#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================================="
echo " СТАРТ ОПТИМИЗИРОВАННОГО ИССЛЕДОВАНИЯ (REFINED ABLATION STUDY)   "
echo "================================================================="

OUT_DIR="science_folder"

# ========================================================================
# ЭТАП 1: Доказательство важности предобработки (Baseline)
# ========================================================================
echo "--- ЭТАП 1: Raw vs Preprocessed ---"
python train.py --exp_name "01_base_raw" --loss "ce" --sampling_mode "none" --data_dir "dataset" --out_dir $OUT_DIR --batch_size 64
python train.py --exp_name "02_base_prep" --loss "ce" --sampling_mode "none" --data_dir "dataset_preprocessed" --out_dir $OUT_DIR --batch_size 64

# ========================================================================
# ЭТАП 2: Стратегии балансировки потока данных (Data-level)
# ========================================================================
echo "--- ЭТАП 2: Data-level Balancing (Samplers) ---"
python train.py --exp_name "03_prep_undersample" --loss "ce" --sampling_mode "undersample" --data_dir "dataset_preprocessed" --out_dir $OUT_DIR --batch_size 64
python train.py --exp_name "04_prep_oversample" --loss "ce" --sampling_mode "oversample" --data_dir "dataset_preprocessed" --out_dir $OUT_DIR --batch_size 64

# ========================================================================
# ЭТАП 3: Стратегии балансировки на уровне градиентов (Algorithm-level)
# ========================================================================
echo "--- ЭТАП 3: Algorithm-level Balancing (Weights & Focal) ---"
python train.py --exp_name "05_prep_weighted_ce" --loss "ce" --use_weights --sampling_mode "none" --data_dir "dataset_preprocessed" --out_dir $OUT_DIR --batch_size 64

python train.py --exp_name "06_prep_focal_g0.5" --loss "focal" --gamma 0.5 --sampling_mode "none" --data_dir "dataset_preprocessed" --out_dir $OUT_DIR --batch_size 64
python train.py --exp_name "07_prep_focal_g1.0" --loss "focal" --gamma 1.0 --sampling_mode "none" --data_dir "dataset_preprocessed" --out_dir $OUT_DIR --batch_size 64
python train.py --exp_name "08_prep_focal_g2.0" --loss "focal" --gamma 2.0 --sampling_mode "none" --data_dir "dataset_preprocessed" --out_dir $OUT_DIR --batch_size 64

# ========================================================================
# ЭТАП 4: Безопасная регуляризация на обычных данных
# ========================================================================
echo "--- ЭТАП 4: Label Smoothing on CE (No GAN) ---"
python train.py --exp_name "09_prep_smooth_ce" --loss "ce" --smoothing 0.1 --sampling_mode "none" --data_dir "dataset_preprocessed" --out_dir $OUT_DIR --batch_size 64

# ========================================================================
# ЭТАП 5: Синтетический баланс (Генеративный подход)
# ========================================================================
echo "--- ЭТАП 5: Generative Balancing (GAN Dataset) ---"
python train.py --exp_name "10_gan_baseline" --loss "ce" --sampling_mode "none" --data_dir "dataset_augmented" --out_dir $OUT_DIR --batch_size 64
python train.py --exp_name "11_gan_smooth" --loss "ce" --smoothing 0.1 --sampling_mode "none" --data_dir "dataset_augmented" --out_dir $OUT_DIR --batch_size 64

# ========================================================================
# ЭТАП 6: Продвинутый SOTA (Representation Learning / SupCon)
# Изолируем влияние: сначала только SupCon, затем SupCon + GAN
# ========================================================================
echo "--- ЭТАП 6: Representation Learning (SupCon) ---"

# 6.1 Изолированный тест SupCon на дисбалансе (Без генерации)
python train.py --exp_name "12_prep_supcon" --loss "ce" --use_supcon --supcon_weight 0.1 --sampling_mode "none" --data_dir "dataset_preprocessed" --out_dir $OUT_DIR --batch_size 64

# 6.2 Ультимативное комбо (SupCon + Идеальный баланс от GAN)
python train.py --exp_name "13_gan_supcon" --loss "ce" --use_supcon --supcon_weight 0.1 --sampling_mode "none" --data_dir "dataset_augmented" --out_dir $OUT_DIR --batch_size 64

# ========================================================================
# ЭТАП 7: Weighted Focal Loss (Комбо: Веса + Сложность)
# ========================================================================
echo "--- ЭТАП 7: Weighted Focal Loss (The Ultimate Balance) ---"

python train.py --exp_name "14_prep_weighted_focal_g0.5" --loss "focal" --gamma 0.5 --use_weights --sampling_mode "none" --data_dir "dataset_preprocessed" --out_dir $OUT_DIR --batch_size 64
python train.py --exp_name "15_prep_weighted_focal_g1.0" --loss "focal" --gamma 1.0 --use_weights --sampling_mode "none" --data_dir "dataset_preprocessed" --out_dir $OUT_DIR --batch_size 64
python train.py --exp_name "16_prep_weighted_focal_g2.0" --loss "focal" --gamma 2.0 --use_weights --sampling_mode "none" --data_dir "dataset_preprocessed" --out_dir $OUT_DIR --batch_size 64

echo "================================================================="
echo " ВСЕ 13 ЭКСПЕРИМЕНТОВ УСПЕШНО ЗАВЕРШЕНЫ!                         "
echo "================================================================="