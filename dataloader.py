import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

class InMemoryImageFolder(Dataset):
    def __init__(self, root_dir, transform=None, img_size=(256, 256)):
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Парсинг структуры папок
        temp_dataset = datasets.ImageFolder(root=root_dir)
        self.classes = temp_dataset.classes
        self.class_to_idx = temp_dataset.class_to_idx
        
        print(f"[INFO] Кэширование {root_dir} в ОЗУ...")
        for img_path, label in tqdm(temp_dataset.imgs, leave=False):
            # Ресайз до кэширования экономит гигабайты памяти
            img = Image.open(img_path).convert("RGB").resize(img_size)
            self.images.append(img)
            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(data_root="dataset", batch_size=64, use_augmentation=False, num_workers=8):
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = base_transform

    # Загрузка в память
    train_set = InMemoryImageFolder(os.path.join(data_root, 'train'), transform=train_transform)
    val_set = InMemoryImageFolder(os.path.join(data_root, 'valid'), transform=base_transform)
    test_set = InMemoryImageFolder(os.path.join(data_root, 'test'), transform=base_transform)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True if num_workers > 0 else False,
        "prefetch_factor": 4 if num_workers > 0 else None
    }

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=12, pin_memory=True, persistent_workers=True,
                              prefetch_factor=4)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=12, pin_memory=True, persistent_workers=True,
                            prefetch_factor=4)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, train_set.classes