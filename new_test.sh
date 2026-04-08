#!/bin/bash

echo "================================================================="
echo " СТАРТ ФИНАЛЬНОГО БЛОКА: SUPERVISED CONTRASTIVE LEARNING "
echo "================================================================="

# 1. Добиваем проверку SupCon на обычных (предобработанных) данных с Focal Loss
# Мы ожидаем, что Focal Loss исправит ошибки классификатора, пока SupCon кластеризует фичи
# echo "--- ЭТАП 7.1: SupCon + Focal Loss (Preprocessed) ---"
# python train.py --name "7_prep_supcon_focal2" --loss "focal" --gamma 2.0 --use_supcon --supcon_weight 0.1 --data_dir "dataset_preprocessed" --out_dir "diploma_results__full" --batch_size 256

# 2. Переносим SupCon на GAN-датасет
# Чистая кросс-энтропия и Focal Loss на синтетике
echo "--- ЭТАП 7.2: SupCon (GAN Augmented Data) ---"
python train.py --name "7_gan_supcon_ce" --loss "ce" --use_supcon --supcon_weight 0.1 --data_dir "dataset_augmented" --out_dir "diploma_results__full" --batch_size 256
python train.py --name "7_gan_supcon_focal2" --loss "focal" --gamma 2.0 --use_supcon --supcon_weight 0.1 --data_dir "dataset_augmented" --out_dir "diploma_results__full" --batch_size 256

echo "================================================================="
echo " ВСЕ ВЫЧИСЛЕНИЯ ДЛЯ ДИПЛОМА ЗАВЕРШЕНЫ! "
echo "================================================================="