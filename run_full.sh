#!/bin/bash

echo "================================================================="
echo " СТАРТ ПОЛНОГО СЕТОЧНОГО ИССЛЕДОВАНИЯ (FULL GRID ABLATION STUDY) "
echo "================================================================="

# ========================================================================
# ЭТАП 1: Базовые модели (Влияние качества данных без аугментаций)
# ========================================================================
echo "--- ЭТАП 1: Baselines (Raw vs Preprocessed) ---"
python train.py --name "1_base_raw_ce" --loss "ce" --data_dir "dataset" --out_dir "diploma_results__full" --batch_size 512
python train.py --name "1_base_prep_ce" --loss "ce" --data_dir "dataset_preprocessed" --out_dir "diploma_results__full" --batch_size 512

# ========================================================================
# ЭТАП 2: Изолированная борьба с дисбалансом (Без аугментаций)
# ========================================================================
echo "--- ЭТАП 2: Class Imbalance Methods (Preprocessed Data) ---"
python train.py --name "2_prep_sampler" --loss "ce" --sampler --data_dir "dataset_preprocessed" --out_dir "diploma_results__full" --batch_size 512
python train.py --name "2_prep_focal_g0.5" --loss "focal" --gamma 0.5 --data_dir "dataset_preprocessed" --out_dir "diploma_results__full" --batch_size 512
python train.py --name "2_prep_focal_g1.0" --loss "focal" --gamma 1.0 --data_dir "dataset_preprocessed" --out_dir "diploma_results__full" --batch_size 512
python train.py --name "2_prep_focal_g2.0" --loss "focal" --gamma 2.0 --data_dir "dataset_preprocessed" --out_dir "diploma_results__full" --batch_size 512

# ========================================================================
# ЭТАП 3: Перебор Аугментаций на ОБЫЧНЫХ данных (Preprocessed)
# Цель: Узнать предел возможностей без генерации
# ========================================================================
echo "--- ЭТАП 3.1: Аугментации + Cross Entropy (Preprocessed) ---"
python train.py --name "3_prep_augGeom_ce" --loss "ce" --augment --data_dir "dataset_preprocessed" --out_dir "diploma_results__full" --batch_size 512
python train.py --name "3_prep_smooth_ce" --loss "ce" --smoothing 0.1 --data_dir "dataset_preprocessed" --out_dir "diploma_results__full" --batch_size 512
python train.py --name "3_prep_mixup_ce" --loss "ce" --mixup --data_dir "dataset_preprocessed" --out_dir "diploma_results__full" --batch_size 512
python train.py --name "3_prep_cutmix_ce" --loss "ce" --cutmix --data_dir "dataset_preprocessed" --out_dir "diploma_results__full" --batch_size 512

echo "--- ЭТАП 3.2: Аугментации + Focal Loss g=2.0 (Preprocessed) ---"
# Проверяем, как штрафы Focal Loss работают вместе с геометрией и сглаживанием
python train.py --name "3_prep_augGeom_focal2" --loss "focal" --gamma 2.0 --augment --data_dir "dataset_preprocessed" --out_dir "diploma_results__full" --batch_size 512
python train.py --name "3_prep_mixup_focal2" --loss "focal" --gamma 2.0 --mixup --data_dir "dataset_preprocessed" --out_dir "diploma_results__full" --batch_size 512
python train.py --name "3_prep_cutmix_focal2" --loss "focal" --gamma 2.0 --cutmix --data_dir "dataset_preprocessed" --out_dir "diploma_results__full" --batch_size 512

# ========================================================================
# ЭТАП 4: Чистый GAN (Влияние синтетических данных)
# ========================================================================
echo "--- ЭТАП 4: Внедрение GAN Dataset (Baseline) ---"
python train.py --name "4_gan_ce" --loss "ce" --data_dir "dataset_augmented" --out_dir "diploma_results__full" --batch_size 512
python train.py --name "4_gan_focal_g2.0" --loss "focal" --gamma 2.0 --data_dir "dataset_augmented" --out_dir "diploma_results__full" --batch_size 512

# ========================================================================
# ЭТАП 5: Перебор Аугментаций на СИНТЕТИЧЕСКИХ данных (GAN)
# Цель: Проверить, работают ли аугментации на уже аугментированных данных
# ========================================================================
echo "--- ЭТАП 5.1: GAN + Аугментации + Cross Entropy ---"
python train.py --name "5_gan_augGeom_ce" --loss "ce" --augment --data_dir "dataset_augmented" --out_dir "diploma_results__full" --batch_size 512
python train.py --name "5_gan_smooth_ce" --loss "ce" --smoothing 0.1 --data_dir "dataset_augmented" --out_dir "diploma_results__full" --batch_size 512
python train.py --name "5_gan_mixup_ce" --loss "ce" --mixup --data_dir "dataset_augmented" --out_dir "diploma_results__full" --batch_size 512
python train.py --name "5_gan_cutmix_ce" --loss "ce" --cutmix --data_dir "dataset_augmented" --out_dir "diploma_results__full" --batch_size 512

echo "--- ЭТАП 5.2: GAN + Аугментации + Focal Loss (Разные Гаммы) ---"
# Тестируем Focal Loss 1.0 (мягкий) на GAN с аугментациями
python train.py --name "5_gan_augGeom_focal1" --loss "focal" --gamma 1.0 --augment --data_dir "dataset_augmented" --out_dir "diploma_results__full" --batch_size 512
python train.py --name "5_gan_mixup_focal1" --loss "focal" --gamma 1.0 --mixup --data_dir "dataset_augmented" --out_dir "diploma_results__full" --batch_size 512

# Тестируем Focal Loss 2.0 (жесткий) на GAN с аугментациями
python train.py --name "5_gan_augGeom_focal2" --loss "focal" --gamma 2.0 --augment --data_dir "dataset_augmented" --out_dir "diploma_results__full" --batch_size 512
python train.py --name "5_gan_mixup_focal2" --loss "focal" --gamma 2.0 --mixup --data_dir "dataset_augmented" --out_dir "diploma_results__full" --batch_size 512

# ========================================================================
# ЭТАП 6: АБСОЛЮТНЫЙ SOTA (Ultimate Combo Grid)
# Скрещиваем ВСЁ: Сгенерированные данные + Геометрия + Сглаживание/MixUp + Перебор Loss
# ========================================================================
echo "--- ЭТАП 6: THE ULTIMATE SOTA GRID (GAN + All Augs) ---"

# Комбо на Cross-Entropy
python train.py --name "6_sota_combo_ce" --loss "ce" --augment --smoothing 0.1 --mixup --cutmix --data_dir "dataset_augmented" --out_dir "diploma_results__full" --batch_size 512

# Комбо на Focal Loss с перебором Гаммы (0.5, 1.0, 2.0)
python train.py --name "6_sota_combo_focal0.5" --loss "focal" --gamma 0.5 --augment --smoothing 0.1 --mixup --cutmix --data_dir "dataset_augmented" --out_dir "diploma_results__full" --batch_size 512
python train.py --name "6_sota_combo_focal1.0" --loss "focal" --gamma 1.0 --augment --smoothing 0.1 --mixup --cutmix --data_dir "dataset_augmented" --out_dir "diploma_results__full" --batch_size 512
python train.py --name "6_sota_combo_focal2.0" --loss "focal" --gamma 2.0 --augment --smoothing 0.1 --mixup --cutmix --data_dir "dataset_augmented" --out_dir "diploma_results__full" --batch_size 512

echo "================================================================="
echo " ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ! ВЫ ГОТОВЫ К ЗАЩИТЕ ДИПЛОМА! "
echo "================================================================="