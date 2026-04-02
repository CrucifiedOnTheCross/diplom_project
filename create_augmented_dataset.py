import os
import subprocess
import shutil
from pathlib import Path

def main():
    src_dir = Path('dataset_preprocessed/train')
    aug_dir = Path('dataset_augmented/train')

    print(f"=========================================================")
    print(f"[*] Шаг 1: Клонирование оригинального датасета...")
    print(f"    Из: {src_dir}")
    print(f"    В:  {aug_dir}")
    # Копируем всё, если папка уже есть - перезаписываем/добавляем
    shutil.copytree(src_dir, aug_dir, dirs_exist_ok=True)
    print(f"[+] Клонирование завершено!")
    
    # 1. Узнаем размер мажоритарного класса в новом датасете
    nv_dir = aug_dir / 'nv'
    target_count = len(list(nv_dir.glob('*.jpg')) + list(nv_dir.glob('*.png')))
    print(f"\n[*] Целевой размер каждого класса: {target_count} изображений")

    networks = {
        'vasc': 'gan_training_runs/vasc_sg3/00001-stylegan2-vasc_dataset_sg3-gpus1-batch32-gamma1/network-snapshot-000200.pkl',
        'df': 'gan_training_runs/df_sg3/00000-stylegan2-df_dataset_sg3-gpus1-batch32-gamma1/network-snapshot-000960.pkl',
        'akiec': 'gan_training_runs/akiec_sg3/00000-stylegan2-akiec_dataset_sg3-gpus1-batch32-gamma1/network-snapshot-001000.pkl',
        'bcc': 'gan_training_runs/bcc_sg3/00000-stylegan2-bcc_dataset_sg3-gpus1-batch32-gamma1/network-snapshot-001000.pkl'
    }

    print(f"\n[*] Шаг 2: Генерация недостающих снимков (GAN)...")
    for cls, network_pkl in networks.items():
        cls_dir = aug_dir / cls
        
        # Проверяем, существует ли файл с весами
        if not Path(network_pkl).exists():
            print(f"[!] Ошибка: Не найден файл весов для {cls} по пути: {network_pkl}")
            print(f"    Пропускаем этот класс.")
            continue

        current_count = len(list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png')))
        
        diff = target_count - current_count
        if diff <= 0:
            print(f"[+] Класс {cls} уже сбалансирован. Пропуск.")
            continue
            
        print(f"\n--- Класс {cls} ---")
        print(f"Нужно сгенерировать: {diff} шт.")
        print(f"Используем веса: {network_pkl}")
        
        seeds_arg = f"0-{diff - 1}"
        cmd = [
            "python", "stylegan3-brecahad/stylegan3/gen_images.py",
            f"--outdir={str(cls_dir)}",
            f"--trunc=0.7",
            f"--seeds={seeds_arg}",
            f"--network={network_pkl}"
        ]
        
        subprocess.run(cmd, check=True)
        print(f"[SUCCESS] Добавлено {diff} синтетических изображений в {cls_dir}!")

    print("\n=========================================================")
    print(" РАСШИРЕННЫЙ ДАТАСЕТ УСПЕШНО СОЗДАН!")
    print(" Теперь у тебя есть два датасета для экспериментов:")
    print(" 1. dataset_preprocessed (Оригинал, с дисбалансом)")
    print(" 2. dataset_augmented    (Расширенный GANом, идеальный баланс)")
    print("=========================================================")

if __name__ == '__main__':
    main()