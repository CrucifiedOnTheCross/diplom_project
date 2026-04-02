import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import v2  # Используем v2 для MixUp и CutMix
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Импорты ваших кастомных модулей
from model import SkinLesionClassifier
from losses import get_loss_function
from metrics import (
    HistoryTracker, calculate_advanced_metrics, 
    plot_training_results, EarlyStopping, save_medical_report
)
from ram_dataset import RAMFullDataset

# ==========================================
# ЭКСТРЕМАЛЬНЫЕ ОПТИМИЗАЦИИ (RTX 5080)
# ==========================================
torch.backends.cudnn.benchmark = True 
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser(description="HAM10000 - Ультимативный тренировочный пайплайн")
    # Базовые настройки
    parser.add_argument('--name', type=str, required=True, help="Имя эксперимента")
    parser.add_argument('--data_dir', type=str, default='dataset', help="Папка с данными")
    parser.add_argument('--out_dir', type=str, default='diploma_results', help="Главная папка для сохранения результатов")
    
    # Гиперпараметры обучения
    parser.add_argument('--batch_size', type=int, default=512, help="Размер батча")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=3, help="Эпохи прогрева головы")
    parser.add_argument('--lr_head', type=float, default=3e-3, help="LR для головы")
    parser.add_argument('--lr_backbone', type=float, default=5e-4, help="Max LR для бэкбона")
    
    # Методы борьбы с дисбалансом
    parser.add_argument('--loss', type=str, choices=['ce', 'focal'], default='ce')
    parser.add_argument('--gamma', type=float, default=2.0, help="Gamma для Focal Loss")
    parser.add_argument('--sampler', action='store_true', help="Включить Oversampling (WeightedRandomSampler)")
    
    # Продвинутые аугментации и регуляризация
    parser.add_argument('--augment', action='store_true', help="Включить базовую GPU-аугментацию (геометрия)")
    parser.add_argument('--smoothing', type=float, default=0.0, help="Label Smoothing (например, 0.1)")
    parser.add_argument('--mixup', action='store_true', help="Включить MixUp аугментацию")
    parser.add_argument('--cutmix', action='store_true', help="Включить CutMix аугментацию")
    
    return parser.parse_args()

def train_model():
    args = parse_args()
    device = torch.device("cuda")
    
    # Создание директории эксперимента
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = Path(args.out_dir) / f"{timestamp}_{args.name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.txt", "w") as f: f.write(str(args))

    print(f"\n[START] {args.name} | Данные: {args.data_dir} | Выход: {exp_dir}")
    
    # ==========================================
    # 1. ЗАГРУЗКА ДАННЫХ
    # ==========================================
    dataset_path = Path(args.data_dir)
    train_dataset = RAMFullDataset(dataset_path / 'train')
    val_dataset = RAMFullDataset(dataset_path / 'valid')

    loader_kwargs = {
        'batch_size': args.batch_size,
        'pin_memory': True,
        'num_workers': 8,
        'persistent_workers': True,
        'prefetch_factor': 2
    }

    if args.sampler:
        print("\n[*] ВНИМАНИЕ: Включен Сэмплер (WeightedRandomSampler)!")
        class_counts = torch.bincount(train_dataset.labels)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[train_dataset.labels].cpu()
        
        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(sample_weights), 
            replacement=True
        )
        train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
    else:
        print("\n[*] Используется стандартное перемешивание (shuffle=True).")
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)

    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # ==========================================
    # 2. ИНИЦИАЛИЗАЦИЯ ФУНКЦИИ ПОТЕРЬ
    # ==========================================
    num_classes = len(train_dataset.classes)
    class_counts = torch.bincount(train_dataset.labels)
    total_samples = len(train_dataset.labels)
    
    dynamic_weights = total_samples / (num_classes * class_counts.float())
    print("\n[*] Динамические веса Alpha (вычислены по тренировочной выборке):")
    for cls_name, weight in zip(train_dataset.classes, dynamic_weights):
        print(f"    - {cls_name}: {weight:.4f}")
    
    if args.loss == 'ce' and args.smoothing > 0:
        print(f"[*] Используется CrossEntropyLoss с Label Smoothing = {args.smoothing}")
        criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing).to(device)
    else:
        criterion = get_loss_function(
            name=args.loss, 
            gamma=args.gamma, 
            class_weights=dynamic_weights if args.loss == 'focal' else None, 
            device=device
        )

    # ==========================================
    # 3. НАСТРОЙКА АУГМЕНТАЦИЙ (GPU)
    # ==========================================
    # Базовая геометрия
    gpu_augment = nn.Sequential(
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
    ).to(device)

    # Продвинутая регуляризация (MixUp / CutMix)
    mixup_cutmix_transforms = []
    if args.mixup:
        mixup_cutmix_transforms.append(v2.MixUp(num_classes=num_classes, alpha=0.2))
    if args.cutmix:
        mixup_cutmix_transforms.append(v2.CutMix(num_classes=num_classes, alpha=1.0))
        
    batch_collator = v2.RandomChoice(mixup_cutmix_transforms) if mixup_cutmix_transforms else None
    if batch_collator:
        print(f"[*] Включены батч-аугментации: MixUp={args.mixup}, CutMix={args.cutmix}")

    # ==========================================
    # 4. ИНИЦИАЛИЗАЦИЯ МОДЕЛИ
    # ==========================================
    model = SkinLesionClassifier(num_classes=num_classes).to(device)
    if hasattr(torch, 'compile'):
        model = torch.compile(model)

    scaler = torch.amp.GradScaler('cuda')
    tracker = HistoryTracker()
    model_save_path = exp_dir / 'best_model.pth'
    early_stopping = EarlyStopping(patience=10, mode='max', save_path=str(model_save_path))

    optimizer = None
    scheduler = None

    # ==========================================
    # 5. ЦИКЛ ОБУЧЕНИЯ
    # ==========================================
    for epoch in range(args.epochs):
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        
        # Управление заморозкой/разморозкой и LR Schedulers
        if epoch == 0:
            print(f"\n[Stage 1] Прогрев полносвязной головы ({args.warmup} эпох)...")
            raw_model.freeze_backbone()
            optimizer = optim.AdamW(raw_model.head.parameters(), lr=args.lr_head, weight_decay=1e-4)
            scheduler = None 
            
        elif epoch == args.warmup:
            print(f"\n[Stage 2] Разморозка бэкбона. Запуск OneCycleLR...")
            raw_model.unfreeze_backbone()
            optimizer = optim.AdamW(model.parameters(), lr=args.lr_backbone, weight_decay=1e-4)
            
            fine_tune_epochs = args.epochs - args.warmup
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.lr_backbone,
                epochs=fine_tune_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.1, 
                anneal_strategy='cos',
                final_div_factor=100.0 
            )

        # ----------------- ТРЕНИРОВКА -----------------
        model.train()
        train_preds, train_labels, running_loss = [], [], 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)

            # 1. Геометрические аугментации
            if args.augment:
                inputs = gpu_augment(inputs)

            # 2. MixUp / CutMix аугментации
            if batch_collator is not None:
                inputs, labels = batch_collator(inputs, labels)

            # 3. Forward Pass
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # 4. Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if scheduler is not None:
                scheduler.step()

            # 5. Сбор метрик
            running_loss += loss.item() * inputs.size(0)
            train_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            
            # Если использовался MixUp/CutMix, labels становятся вероятностями (2D тензор).
            # Для расчета обычного Accuracy восстанавливаем главный класс через argmax.
            if labels.ndim == 2:
                train_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())
            else:
                train_labels.extend(labels.cpu().numpy())

        # ----------------- ВАЛИДАЦИЯ -----------------
        model.eval()
        val_preds, val_labels, val_probs, val_loss = [], [], [], 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                val_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Расчет и логирование
        metrics_step = calculate_advanced_metrics(val_labels, val_preds, np.array(val_probs))
        tracker.update(running_loss/len(train_dataset), val_loss/len(val_dataset),
                       accuracy_score(train_labels, train_preds), accuracy_score(val_labels, val_preds),
                       balanced_accuracy_score(train_labels, train_preds), balanced_accuracy_score(val_labels, val_preds))

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Loss: {val_loss/len(val_dataset):.4f} | B-Acc: {balanced_accuracy_score(val_labels, val_preds):.4f} | MCC: {metrics_step['MCC']:.4f} | LR: {current_lr:.2e}")

        # Early Stopping по метрике MCC
        early_stopping(metrics_step['MCC'], raw_model)
        if early_stopping.early_stop: 
            print(f"\n[!] Сработал Early Stopping. Обучение прервано.")
            break

    # ==========================================
    # 6. ФИНАЛИЗАЦИЯ И МЕДИЦИНСКИЙ ОТЧЕТ
    # ==========================================
    print(f"\n[*] Генерация финального медицинского отчета...")
    raw_model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model.eval()
    
    final_preds, final_labels, final_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
            final_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            final_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            final_labels.extend(labels.cpu().numpy())

    final_metrics = calculate_advanced_metrics(final_labels, final_preds, np.array(final_probs))
    save_medical_report(exp_dir, final_labels, final_preds, np.array(final_probs), final_metrics)
    plot_training_results(tracker, final_labels, final_preds, save_path=str(exp_dir / 'training_results.png'))
    
    print(f"\n[SUCCESS] Эксперимент завершен!")
    print(f"Результаты лежат в: {exp_dir}")

if __name__ == '__main__':
    train_model()