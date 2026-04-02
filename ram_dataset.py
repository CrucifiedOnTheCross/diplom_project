import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

class RAMFullDataset(Dataset):
    """
    Сверхбыстрый датасет: хранит все изображения 256x256 в оперативной памяти (ОЗУ).
    Тензоры переносятся на GPU только во время формирования батча (через pin_memory).
    """
    def __init__(self, folder_path):
        self.dataset = datasets.ImageFolder(folder_path)
        
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.images, self.labels = [], []
        
        print(f"[*] Перенос {folder_path} в ОЗУ (256x256)... Это займет ~35 ГБ RAM")
        for path, label in tqdm(self.dataset.imgs, desc="Disk -> RAM"):
            with Image.open(path) as img:
                # Сохраняем тензор на CPU (в ОЗУ)
                tensor_img = self.preprocess(img.convert('RGB'))
                self.images.append(tensor_img)
            self.labels.append(torch.tensor(label, dtype=torch.long))
            
        self.classes = self.dataset.classes
        
        # Объединяем списки в единые огромные тензоры в ОЗУ
        self.images = torch.stack(self.images)
        self.labels = torch.stack(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)