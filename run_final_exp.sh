#!/bin/bash

# Останавливать выполнение при ошибке
set -e

echo "========================================================="
echo " Запуск ФИНАЛЬНОЙ серии экспериментов (GAN Augmented Data)"
echo " Папка: dataset_augmented | ConvNeXt-Tiny | OneCycleLR"
echo " Цель: Доказать превосходство генеративной балансировки"
echo "========================================================="

# БЛОК 1: Идеальный бейзлайн (Сбалансированные данные)
# Цель: Посмотреть, как самая обычная кросс-энтропия без костылей 
# работает на идеально ровных классах, дополненных GAN-ом.
echo "-> [1/2] Запуск Cross-Entropy на сбалансированных данных..."
python train.py --name "gan_aug_ce" --loss ce --data_dir "dataset_augmented" --batch_size 512
python train.py --name "gan_aug_ce_aug" --loss ce --augment --data_dir "dataset_augmented" --batch_size 512

# ВНИМАНИЕ: Блок со --sampler УДАЛЕН умышленно! 
# Датасет сбалансирован физически. Алгоритмический Oversampling 
# больше не нужен и будет только тормозить обучение.

# БЛОК 2: Хард-майнинг (Focal Loss на ровных данных)
# Цель: Проверить, поможет ли Focal Loss выжимать максимум из 
# сложных/пограничных примеров, когда проблемы дисбаланса больше нет.
echo "-> [2/2] Запуск Focal Loss (Hard Example Mining)..."
python train.py --name "gan_aug_focal_g1.0_aug" --loss focal --gamma 1.0 --augment --data_dir "dataset_augmented" --batch_size 512
python train.py --name "gan_aug_focal_g2.0_aug" --loss focal --gamma 2.0 --augment --data_dir "dataset_augmented" --batch_size 512

echo "========================================================="
echo "Триумф! Эксперименты на сбалансированном GAN-датасете завершены."
echo "Запустите collect_results.py для финального сравнения с Clean Data!"
echo "========================================================="