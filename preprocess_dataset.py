import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def apply_color_constancy(img):
    """Алгоритм 'Серый мир' для нормализации цвета"""
    result_gw = np.zeros_like(img, dtype=np.float32)
    avg_r = np.mean(img[:,:,0])
    avg_g = np.mean(img[:,:,1])
    avg_b = np.mean(img[:,:,2])
    
    # Защита от деления на ноль
    avg = (avg_r + avg_g + avg_b) / 3.0
    if avg_r == 0 or avg_g == 0 or avg_b == 0:
        return img
        
    result_gw[:,:,0] = np.clip(img[:,:,0] * (avg / avg_r), 0, 255)
    result_gw[:,:,1] = np.clip(img[:,:,1] * (avg / avg_g), 0, 255)
    result_gw[:,:,2] = np.clip(img[:,:,2] * (avg / avg_b), 0, 255)
    return result_gw.astype(np.uint8)

def remove_hair(img):
    """Морфологический Black-Hat + Inpainting (Аналог DullRazor)"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Ядро крестом отлично цепляет длинные тонкие структуры (волосы)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Порог 15 отсекает шумы, оставляя только явные темные волосы
    _, hair_mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
    
    # Закрашивание. Telea работает чуть быстрее Navier-Stokes
    img_hairless = cv2.inpaint(img, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return img_hairless

def process_single_image(args):
    """Обрабатывает одну картинку и сохраняет её"""
    src_path, dst_path = args
    
    # Пропускаем, если уже обработано (полезно при сбоях)
    if os.path.exists(dst_path):
        return
        
    img = cv2.imread(str(src_path))
    if img is None:
        return
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Цветовая константность
    img = apply_color_constancy(img)
    
    # 2. Удаление артефактов и волос
    img = remove_hair(img)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(dst_path), img)

def main():
    src_dir = Path('dataset')
    dst_dir = Path('dataset_preprocessed')
    
    if not src_dir.exists():
        print("Исходная папка dataset/ не найдена!")
        return
        
    print(f"[*] Начинаем глубокую предобработку датасета...")
    print(f"[*] Исходная папка: {src_dir}")
    print(f"[*] Целевая папка:  {dst_dir}")
    
    tasks = []
    
    # Создаем зеркальную структуру папок и собираем задачи
    for root, dirs, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        target_root = dst_dir / rel_path
        target_root.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = Path(root) / file
                dst_path = target_root / file
                tasks.append((src_path, dst_path))
                
    print(f"[*] Найдено {len(tasks)} изображений. Запуск пула потоков...")
    
    # Запускаем обработку в несколько потоков
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(process_single_image, tasks), total=len(tasks), desc="Обработка"))
        
    print(f"\n[SUCCESS] Датасет успешно обработан и сохранен в {dst_dir}")

if __name__ == '__main__':
    main()