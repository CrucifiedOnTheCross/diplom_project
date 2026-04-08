#!/bin/bash
set -e

# Указываем все 4 класса, папки которых есть на твоем скриншоте
CLASSES=("akiec" "bcc" "vasc")

# ВАЖНО: Целевое количество kimg. 
# Если прошлый раз было 1000, ставь 2000 или 3000.
TARGET_KIMG=2000 

for CLS in "${CLASSES[@]}"; do
    echo "========================================================="
    echo " ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ GAN ДЛЯ КЛАССА: $CLS"
    echo "========================================================="
    
    # 1. Автоматический поиск самого свежего .pkl чекпоинта в папке класса
    # (StyleGAN создает подпапки вида 00000-..., поэтому ищем внутри них)
    LATEST_PKL=$(ls -t gan_training_runs/${CLS}_sg3/*/*.pkl 2>/dev/null | head -n 1)

    # Проверка, найден ли файл
    if [ -z "$LATEST_PKL" ]; then
        echo "[ОШИБКА] Не найден файл чекпоинта .pkl для класса $CLS! Пропускаем..."
        continue
    fi

    echo "[*] Найден последний чекпоинт: $LATEST_PKL"
    echo "[*] Запуск дообучения до достижения $TARGET_KIMG kimg..."

    # 2. Запуск дообучения с флагом --resume
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
        --kimg=$TARGET_KIMG \
        --resume=$LATEST_PKL

    echo "Дообучение для $CLS успешно запущено/завершено!"
    echo "---------------------------------------------------------"
done

echo "=== ВСЕ GAN УСПЕШНО ДООБУЧЕНЫ ==="