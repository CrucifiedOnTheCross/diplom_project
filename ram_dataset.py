import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

class HAMRAMDataset(Dataset):
    """
    Финальная версия датасета: 
    - uint8 хранение (экономия RAM в 4 раза)
    - Бинарный кэш .pt (загрузка за секунды)
    - Многопоточное чтение
    - Поддержка n_views для SupCon и стандартного режима
    - Поддержка выбора типа аугментации (aug_type: 1 или 2)
    """
    def __init__(self, folder_path, mode='train', n_views=1, cache_name="ham_cache_u8.pt", aug_type=1):
        self.folder_path = Path(folder_path)
        self.dataset = datasets.ImageFolder(folder_path)
        self.classes = self.dataset.classes
        self.mode = mode
        self.n_views = n_views
        self.aug_type = aug_type
        
        # 1. Базовая трансформация (только ресайз и перевод в тензор uint8)
        self.base_transform = v2.Compose([
            v2.Resize((256, 256), antialias=True),
            v2.PILToTensor() 
        ])

        # Стандартные константы нормализации
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # 2. Динамические аугментации (применяются в ОЗУ "на лету")
        if mode == 'train':
            if self.aug_type == 1:
                # Type 1: Старая, более агрессивная аугментация
                self.transform = v2.Compose([
                    v2.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0), antialias=True),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomVerticalFlip(p=0.5),
                    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=mean, std=std)
                ])
            elif self.aug_type == 2:
                # Type 2: Новая, продвинутая и безопасная аугментация
                self.transform = v2.Compose([
                    v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomVerticalFlip(p=0.5),
                    v2.RandomRotation(15),
                    
                    # Легкое изменение цвета с вероятностью 30%
                    v2.RandomApply([
                        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05)
                    ], p=0.3),
                    
                    # Легкое размытие (имитация расфокуса камеры) с вероятностью 20%
                    v2.RandomApply([
                        v2.GaussianBlur(kernel_size=3)
                    ], p=0.2),
                    
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=mean, std=std)
                ])
            else:
                raise ValueError(f"Неизвестный тип аугментации: {self.aug_type}")
        else: # valid/test
            self.transform = v2.Compose([
                v2.Resize((224, 224), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std)
            ])

        # --- МАГИЯ КЭШИРОВАНИЯ ---
        cache_path = self.folder_path / cache_name
        if cache_path.exists():
            print(f"[*] Загрузка бинарного кэша: {cache_path}")
            data = torch.load(cache_path, weights_only=True)
            self.images = data['images']
            self.labels = data['labels']
        else:
            self._load_and_cache(cache_path)

    def _load_and_cache(self, cache_path):
        print(f"[*] Кэш не найден. Чтение {len(self.dataset.imgs)} файлов в {os.cpu_count()} потоков...")
        
        def process_one(item):
            img_path, label = item
            with Image.open(img_path) as img:
                return self.base_transform(img.convert('RGB')), torch.tensor(label)

        num_workers = min(32, (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=num_workers) as exe:
            results = list(tqdm(exe.map(process_one, self.dataset.imgs), 
                                total=len(self.dataset.imgs), desc="Disk -> RAM"))

        images_list, labels_list = zip(*results)
        self.images = torch.stack(images_list)
        self.labels = torch.stack(labels_list)

        print(f"[*] Сохранение кэша в {cache_path}...")
        torch.save({'images': self.images, 'labels': self.labels}, cache_path)

    def __getitem__(self, index):
        img_u8 = self.images[index]
        label = self.labels[index]

        if self.n_views > 1:
            # Режим Contrastive (SupCon)
            views = [self.transform(img_u8) for _ in range(self.n_views)]
            return views, label
        else:
            # Стандартный режим (Baseline)
            return self.transform(img_u8), label

    def __len__(self):
        return len(self.images)