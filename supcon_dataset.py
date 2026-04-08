import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

class SupConRAMDataset(Dataset):
    def __init__(self, folder_path, n_views=2, cache_name="cache_uint8_256.pt"):
        self.dataset = datasets.ImageFolder(folder_path)
        self.n_views = n_views
        self.classes = self.dataset.classes
        
        # ШАГ 1: Легкая базовая трансформация (в uint8 для ОЗУ)
        self.base_transform = v2.Compose([
            v2.Resize((256, 256), antialias=True),
            v2.PILToTensor() 
        ])

        # ШАГ 2: Тяжелые аугментации (будут применяться "на лету" в __getitem__)
        self.contrastive_transform = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        cache_path = Path(folder_path) / cache_name
        
        # =========================================================
        # МАГИЯ УСКОРЕНИЯ: ЗАГРУЗКА ИЗ БИНАРНОГО КЭША
        # =========================================================
        if cache_path.exists():
            print(f"[*] Найден кэш датасета! Мгновенная загрузка из {cache_path}...")
            # Загружаем готовые тензоры напрямую в память
            cache_data = torch.load(cache_path, weights_only=True)
            self.images = cache_data['images']
            self.labels = cache_data['labels']
            print(f"[*] Готово! Загружено изображений: {len(self.images)}")
            return

        # =========================================================
        # ЕСЛИ КЭША НЕТ: МНОГОПОТОЧНОЕ ЧТЕНИЕ С ДИСКА
        # =========================================================
        print(f"[*] Кэш не найден. Запуск многопоточного чтения в ОЗУ...")
        
        # Функция для обработки одной картинки (будет работать в потоках)
        def process_image(item):
            path, label = item
            with Image.open(path) as img:
                # Конвертируем, ресайзим и сразу переводим в uint8 тензор
                tensor = self.base_transform(img.convert('RGB'))
            return tensor, torch.tensor(label, dtype=torch.long)

        # Оптимальное количество потоков для I/O операций (диск)
        num_workers = min(32, (os.cpu_count() or 1) + 4)
        
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Распределяем задачи по потокам и рисуем общий прогресс-бар
            for result in tqdm(executor.map(process_image, self.dataset.imgs), 
                               total=len(self.dataset.imgs), 
                               desc="Multi-Thread Disk -> RAM"):
                results.append(result)
        
        # Разделяем результаты на картинки и метки
        images_list, labels_list = zip(*results)
        
        print("[*] Сборка финальных тензоров...")
        self.images = torch.stack(images_list)
        self.labels = torch.stack(labels_list)

        # =========================================================
        # СОХРАНЕНИЕ КЭША НА БУДУЩЕЕ
        # =========================================================
        print(f"[*] Сохранение бинарного кэша ({cache_path}) для будущих запусков...")
        # Сохраняем словарь с тензорами. В следующий раз это загрузится за секунды.
        torch.save({'images': self.images, 'labels': self.labels}, cache_path)
        print("[*] Инициализация датасета успешно завершена!")

    def __getitem__(self, index):
        base_img = self.images[index]
        views = [self.contrastive_transform(base_img) for _ in range(self.n_views)]
        return views, self.labels[index]

    def __len__(self):
        return len(self.images)