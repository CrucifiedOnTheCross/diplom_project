#!/bin/bash

echo "=== СТАРТ ЦИКЛА ЭКСПЕРИМЕНТОВ ДЛЯ ДИПЛОМА ==="

# Этап 1: Оценка качества данных
# echo "--- Запуск Этапа 1: Базовые модели (Baselines) ---"
# python train.py --name "baseline_raw_ce" --loss "ce" --data_dir "dataset" --out_dir "diploma_results" --batch_size 512
# python train.py --name "preprocessed_ce" --loss "ce" --data_dir "dataset_preprocessed" --out_dir "diploma_results" --batch_size 512

# Этап 2: Алгоритмическая борьба с дисбалансом (Сэмплер и перебор Focal Loss)
echo "--- Запуск Этапа 2: Sampler и Focal Loss (gamma = 0.5, 1.0, 2.0) ---"
python train.py --name "preprocessed_sampler" --loss "ce" --sampler --data_dir "dataset_preprocessed" --out_dir "diploma_results" --batch_size 512

# Тестируем разную жесткость Focal Loss на предобработанных данных
python train.py --name "preprocessed_focal_g0.5" --loss "focal" --gamma 0.5 --data_dir "dataset_preprocessed" --out_dir "diploma_results" --batch_size 512
python train.py --name "preprocessed_focal_g1.0" --loss "focal" --gamma 1.0 --data_dir "dataset_preprocessed" --out_dir "diploma_results" --batch_size 512
python train.py --name "preprocessed_focal_g2.0" --loss "focal" --gamma 2.0 --data_dir "dataset_preprocessed" --out_dir "diploma_results" --batch_size 512

# Этап 3: Генеративная аугментация (GAN)
echo "--- Запуск Этапа 3: GAN Dataset (Чистая Кросс-Энтропия) ---"
python train.py --name "gan_ce" --loss "ce" --data_dir "dataset_augmented" --out_dir "diploma_results" --batch_size 512

# Этап 4: Финальные модели (GAN + Augmentations + Focal)
echo "--- Запуск Этапа 4: Совмещение GAN, аугментаций и Focal Loss ---"
python train.py --name "gan_aug_ce" --loss "ce" --augment --data_dir "dataset_augmented" --out_dir "diploma_results" --batch_size 512

# Смотрим, поможет ли Focal Loss на уже сбалансированном датасете
python train.py --name "gan_aug_focal_g1.0" --loss "focal" --gamma 1.0 --augment --data_dir "dataset_augmented" --out_dir "diploma_results" --batch_size 512
python train.py --name "gan_aug_focal_g2.0" --loss "focal" --gamma 2.0 --augment --data_dir "dataset_augmented" --out_dir "diploma_results" --batch_size 512

echo "=== ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ! РЕЗУЛЬТАТЫ В ПАПКЕ diploma_results ==="