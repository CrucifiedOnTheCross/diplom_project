import cv2
import os
from pathlib import Path
from tqdm import tqdm

def prepare_for_gan(base_src_dir, base_out_dir, classes, size=(256, 256)):
    base_src = Path(base_src_dir)
    base_out = Path(base_out_dir)

    for cls in classes:
        src_path = base_src / cls
        out_path = base_out / f"{cls}_raw"
        
        if not src_path.exists():
            print(f"[!] Папка {src_path} не найдена. Пропускаем...")
            continue
            
        out_path.mkdir(parents=True, exist_ok=True)

        images = list(src_path.glob('*.jpg')) + list(src_path.glob('*.png'))
        print(f"\n[*] Класс '{cls}': Найдено {len(images)} изображений для подготовки...")

        for img_path in tqdm(images, desc=f"Обработка {cls} (256x256)"):
            img = cv2.imread(str(img_path))
            if img is None: continue

            # 1. Обрезка по центру до идеального квадрата
            h, w = img.shape[:2]
            min_dim = min(h, w)
            start_x = w//2 - min_dim//2
            start_y = h//2 - min_dim//2
            cropped = img[start_y:start_y+min_dim, start_x:start_x+min_dim]

            # 2. Ресайз до 256x256
            resized = cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)

            # 3. Сохранение в PNG без потери качества (ОБЯЗАТЕЛЬНО ДЛЯ GAN!)
            cv2.imwrite(str(out_path / f"{img_path.stem}.png"), resized)

        print(f"[+] Класс '{cls}' успешно обработан! Сохранено в: {out_path}")

if __name__ == '__main__':
    # Список классов, для которых мы хотим натренировать GAN
    # nv (невус) - пропускаем, их и так 6700 штук
    # mel и bkl (около 1100 штук) - можно добавить, но приоритет у самых редких
    minority_classes = ['akiec', 'bcc', 'df', 'vasc']
    
    print("=========================================================")
    print(" Подготовка данных для StyleGAN2-ADA (NVIDIA)")
    print("=========================================================")
    
    prepare_for_gan(
        base_src_dir='dataset_preprocessed/train',
        base_out_dir='gan_data',
        classes=minority_classes
    )