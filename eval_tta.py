import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Импорты ваших модулей
from model import SkinLesionClassifier
from ram_dataset import RAMFullDataset
from metrics import calculate_advanced_metrics, save_medical_report

# Оптимизации для инференса
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser(description="Инференс с использованием Test-Time Augmentation (TTA)")
    parser.add_argument('--exp_dir', type=str, required=True, help="Путь к папке эксперимента (где лежит best_model.pth)")
    parser.add_argument('--data_dir', type=str, default='dataset_preprocessed', help="Папка с данными валидации")
    parser.add_argument('--batch_size', type=int, default=64, help="Размер батча (меньше, чем при трейне, т.к. TTA умножает батч)")
    return parser.parse_args()

def generate_tta_views(inputs):
    """
    Генерирует 6 различных взглядов (views) на батч изображений прямо на GPU.
    Мы используем только безопасные для дерматоскопии геом. трансформации.
    """
    views = [
        inputs,                                       # 1. Оригинал
        torch.flip(inputs, dims=[3]),                 # 2. Горизонтальное отражение
        torch.flip(inputs, dims=[2]),                 # 3. Вертикальное отражение
        torch.rot90(inputs, k=1, dims=[2, 3]),        # 4. Поворот на 90 градусов
        torch.rot90(inputs, k=2, dims=[2, 3]),        # 5. Поворот на 180 градусов
        torch.rot90(inputs, k=3, dims=[2, 3])         # 6. Поворот на 270 градусов
    ]
    return views

def run_tta_inference():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_path = Path(args.exp_dir)
    model_path = exp_path / 'best_model.pth'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")

    print(f"\n[START] Test-Time Augmentation Evaluation")
    print(f"[*] Эксперимент: {exp_path.name}")
    
    # 1. Загрузка данных (Валидация)
    dataset_path = Path(args.data_dir)
    val_dataset = RAMFullDataset(dataset_path / 'valid')
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )
    num_classes = len(val_dataset.classes)

    # 2. Инициализация и загрузка модели
    model = SkinLesionClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    if hasattr(torch, 'compile'):
        model = torch.compile(model)

    final_preds, final_labels, final_probs = [], [], []

    print(f"[*] Запуск инференса (6 TTA views на каждое изображение)...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="TTA Inference"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = inputs.size(0)
            
            # Накопитель вероятностей для текущего батча
            # Форма: [batch_size, num_classes]
            avg_probs = torch.zeros(batch_size, num_classes, device=device)
            
            # Получаем 6 вариантов батча
            tta_views = generate_tta_views(inputs)
            
            with torch.amp.autocast('cuda'):
                for view in tta_views:
                    outputs = model(view)
                    # Суммируем Softmax вероятности
                    probs = F.softmax(outputs, dim=1)
                    avg_probs += probs
                    
            # Усредняем вероятности
            avg_probs = avg_probs / len(tta_views)
            
            # Получаем финальные предсказания
            preds = torch.argmax(avg_probs, dim=1)
            
            final_probs.extend(avg_probs.cpu().numpy())
            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())

    # 3. Расчет и сохранение метрик
    print("\n[*] Подсчет метрик с учетом TTA...")
    final_metrics = calculate_advanced_metrics(final_labels, final_preds, np.array(final_probs))
    
    # Выводим главное на экран
    b_acc = balanced_accuracy_score(final_labels, final_preds)
    mcc = final_metrics['MCC']
    print("\n" + "="*40)
    print(" РЕЗУЛЬТАТЫ TEST-TIME AUGMENTATION (TTA)")
    print("="*40)
    print(f"Balanced Accuracy: {b_acc:.4f}")
    print(f"MCC:               {mcc:.4f}")
    print(f"Sensitivity:       {final_metrics.get('Sensitivity', 0):.4f}")
    print(f"Specificity:       {final_metrics.get('Specificity', 0):.4f}")
    print("="*40)
    
    # Сохраняем отдельный отчет
    save_medical_report(exp_path, final_labels, final_preds, np.array(final_probs), final_metrics, filename_prefix="TTA_")
    print(f"\n[SUCCESS] Отчет сохранен в папку эксперимента с префиксом 'TTA_'")

if __name__ == '__main__':
    run_tta_inference()