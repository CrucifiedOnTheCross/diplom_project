#!/bin/bash

# Останавливать выполнение при ошибке
set -e

echo "========================================================="
echo " Запуск ЭЛИТНОЙ серии экспериментов (Clean Data)"
echo " Папка: dataset_preprocessed | ConvNeXt-Tiny | OneCycleLR"
echo "========================================================="

# БЛОК 1: Влияние чистых данных на базовую модель
# Цель: Показать, насколько очистка от волос и выравнивание цвета 
# помогает самой обычной кросс-энтропии.
echo "-> [1/3] Запуск чистых Baseline-экспериментов..."
python train.py --name "clean_baseline_ce" --loss ce --data_dir "dataset_preprocessed" --batch_size 512
python train.py --name "clean_baseline_ce_aug" --loss ce --augment --data_dir "dataset_preprocessed" --batch_size 512

# БЛОК 2: Влияние чистых данных на Сэмплер
# Цель: Проверить, перестанет ли Сэмплер переобучаться, если 
# дубликаты картинок будут очищены от шума.
echo "-> [2/3] Запуск чистой балансировки данных (Oversampling)..."
python train.py --name "clean_ce_sampler_aug" --loss ce --augment --sampler --data_dir "dataset_preprocessed" --batch_size 512

# БЛОК 3: Битва Титанов (Clean Data + Focal Loss)
# Цель: Найти абсолютного чемпиона для диплома. Мы берем две лучшие 
# гаммы из прошлых тестов (1.0 для мин. ошибки меланомы и 2.0 для макс. B-Acc).
echo "-> [3/3] Запуск финальных моделей-чемпионов (Focal Loss)..."
python train.py --name "clean_focal_g1.0_aug" --loss focal --gamma 1.0 --augment --data_dir "dataset_preprocessed" --batch_size 512
python train.py --name "clean_focal_g2.0_aug" --loss focal --gamma 2.0 --augment --data_dir "dataset_preprocessed" --batch_size 512

echo "========================================================="
echo "Финал! Все 5 экспериментов на чистых данных завершены."
echo "Запустите collect_results.py для обновления общей таблицы."