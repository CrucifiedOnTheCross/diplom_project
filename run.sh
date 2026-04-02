#!/bin/bash

# Останавливать выполнение при ошибке в любой команде
set -e

echo "========================================================="
echo " Запуск ПОЛНОЙ серии экспериментов (10 стадий)"
echo " Датасет: HAM10000 | Модель: ConvNeXt-Tiny | ОЗУ VRAM"
echo "========================================================="

# БЛОК 1: Отсутствие балансировки (Базовый уровень)
echo "-> [1/4] Запуск базовых экспериментов (Cross-Entropy)..."
python train.py --name "baseline_ce" --loss ce --batch_size 512
python train.py --name "baseline_ce_aug" --loss ce --augment --batch_size 512

# БЛОК 2: Балансировка на уровне АЛГОРИТМА (Focal Loss)
echo "-> [2/4] Запуск алгоритмической балансировки (Focal Loss + Dynamic Alpha)..."
python train.py --name "focal_g0.5_aug" --loss focal --gamma 0.5 --augment --batch_size 512
python train.py --name "focal_g1.0_aug" --loss focal --gamma 1.0 --augment --batch_size 512
python train.py --name "focal_g2.0_aug" --loss focal --gamma 2.0 --augment --batch_size 512

# БЛОК 3: Балансировка на уровне ДАННЫХ (Weighted Random Sampler)
echo "-> [3/4] Запуск балансировки датасета (Oversampling)..."
# Сравниваем чистый сэмплер с чистым baseline
python train.py --name "ce_sampler" --loss ce --sampler --batch_size 512
# Сравниваем сэмплер + аугментацию с Focal Loss
python train.py --name "ce_sampler_aug" --loss ce --augment --sampler --batch_size 512

# БЛОК 4: Двойная балансировка (Сэмплер + Focal Loss)
# Ожидаем гипер-чувствительность к редким классам и возможное падение Specificity
echo "-> [4/4] Запуск экстремальной 'двойной' балансировки..."
python train.py --name "double_focal_g0.5_aug" --loss focal --gamma 0.5 --augment --sampler --batch_size 512
python train.py --name "double_focal_g1.0_aug" --loss focal --gamma 1.0 --augment --sampler --batch_size 512
python train.py --name "double_focal_g2.0_aug" --loss focal --gamma 2.0 --augment --sampler --batch_size 512

echo "========================================================="
echo "Все 10 экспериментов успешно завершены!"
echo "Результаты сохранены в папке experiments/"