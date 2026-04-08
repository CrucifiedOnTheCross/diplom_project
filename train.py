import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef

# Импорты из твоей кодовой базы
from model import JointSkinLesionClassifier
from ram_dataset import HAMRAMDataset 
from losses import get_loss_function
from metrics import HistoryTracker, EarlyStopping, calculate_advanced_metrics, save_medical_report, plot_training_results

def parse_args():
    parser = argparse.ArgumentParser(description="Ablation Study: Обучение моделей HAM10000")
    parser.add_argument('--exp_name', type=str, required=True, help="Имя эксперимента")
    parser.add_argument('--data_dir', type=str, default='dataset_preprocessed', help="Путь к данным")
    parser.add_argument('--out_dir', type=str, default='science_folder', help="Папка для результатов")
    parser.add_argument('--epochs', type=int, default=50, help="Кол-во эпох")
    
    # Настройки батча и оптимизации памяти
    parser.add_argument('--batch_size', type=int, default=8, help="Физический батч (рекомендуется 8-16)")
    parser.add_argument('--accumulation_steps', type=int, default=4, help="Шаги аккумуляции (8*4=32)")
    parser.add_argument('--lr', type=float, default=1e-4, help="Скорость обучения")
    
    # Конфигурация лоссов и балансировки
    parser.add_argument('--loss', type=str, choices=['ce', 'focal'], default='ce', help="Функция потерь")
    parser.add_argument('--gamma', type=float, default=2.0, help="Gamma для Focal Loss")
    parser.add_argument('--smoothing', type=float, default=0.0, help="Label Smoothing")
    parser.add_argument('--use_weights', action='store_true', help="Включить веса классов")
    parser.add_argument('--sampling_mode', type=str, choices=['none', 'oversample', 'undersample'], default='none')
    
    # Контрастивное обучение
    parser.add_argument('--use_supcon', action='store_true', help="Включить SupCon")
    parser.add_argument('--supcon_weight', type=float, default=0.1, help="Вес SupCon лосса")
    
    return parser.parse_args()

def get_sampler_and_weights(dataset, mode='oversample'):
    labels = dataset.labels.numpy()
    class_counts = np.bincount(labels)
    num_classes = len(class_counts)
    total_samples = len(labels)
    
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights * total_samples / num_classes
    sample_weights = class_weights[labels]
    
    if mode == 'oversample':
        target_num_samples = int(np.max(class_counts) * num_classes)
    elif mode == 'undersample':
        target_num_samples = int(np.min(class_counts) * num_classes)
    else:
        target_num_samples = total_samples

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=target_num_samples, replacement=True)
    return sampler, class_weights

def main():
    args = parse_args()
    
    # --- ГЛОБАЛЬНЫЕ ОПТИМИЗАЦИИ ---
    torch.backends.cudnn.benchmark = True 
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_dir = Path(args.out_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Логируем конфиг в файл
    with open(exp_dir / 'config.txt', 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    print(f"\n[*] Эксперимент: {args.exp_name}")
    print(f"[*] Эффективный батч: {args.batch_size * args.accumulation_steps}")

    # --- ПОДГОТОВКА ДАННЫХ (RAM + Многопоточность) ---
    train_views = 2 if args.use_supcon else 1
    train_dataset = HAMRAMDataset(Path(args.data_dir) / 'train', mode='train', n_views=train_views)
    val_dataset = HAMRAMDataset(Path(args.data_dir) / 'valid', mode='valid', n_views=1)
    
    sampler, class_weights = None, None
    if args.sampling_mode != 'none' or args.use_weights:
        func_mode = args.sampling_mode if args.sampling_mode != 'none' else 'oversample'
        _sampler, _class_weights = get_sampler_and_weights(train_dataset, mode=func_mode)
        sampler = _sampler if args.sampling_mode != 'none' else None
        class_weights = _class_weights if args.use_weights else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=8, pin_memory=True, 
        persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # --- МОДЕЛЬ (С каналами-последними и JIT-компиляцией) ---
    model = JointSkinLesionClassifier(num_classes=len(train_dataset.classes)).to(device, memory_format=torch.channels_last)
    
    try:
        model = torch.compile(model)
        print("[*] Модель оптимизирована через torch.compile.")
    except Exception as e:
        print(f"[*] JIT-компиляция пропущена: {e}")
    
    # --- ЛОССЫ И ОПТИМИЗАТОР ---
    criterion = get_loss_function(
        name=args.loss, gamma=args.gamma, class_weights=class_weights, 
        device=device, label_smoothing=args.smoothing
    )
    supcon_criterion = get_loss_function(name='supcon', device=device) if args.use_supcon else None
    
    # Используем fused=True для ускорения шага AdamW на RTX 5080
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, fused=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    tracker = HistoryTracker(exp_dir=exp_dir)
    early_stopping = EarlyStopping(patience=7, mode='max', save_path=str(exp_dir / 'best_model.pth'))
    scaler = torch.amp.GradScaler('cuda')

    # --- ЦИКЛ ОБУЧЕНИЯ ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_labels, train_preds = 0.0, [], []
        optimizer.zero_grad(set_to_none=True) # Оптимизация сброса градиентов
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        
        for i, (inputs, labels) in enumerate(pbar):
            # Mixed Precision Forward
            with torch.amp.autocast('cuda'):
                if args.use_supcon:
                    inputs = torch.cat(inputs, dim=0).to(device, non_blocking=True, memory_format=torch.channels_last)
                    labels = torch.cat([labels, labels], dim=0).to(device, non_blocking=True)
                    logits, embeddings = model(inputs, return_embeddings=True)
                    loss = (criterion(logits, labels) + args.supcon_weight * supcon_criterion(embeddings, labels)) / args.accumulation_steps
                else:
                    inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
                    labels = labels.to(device, non_blocking=True)
                    logits = model(inputs, return_embeddings=False)
                    loss = criterion(logits, labels) / args.accumulation_steps

            scaler.scale(loss).backward()
            
            # Шаг аккумуляции
            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * args.accumulation_steps
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            if i % 10 == 0:
                pbar.set_postfix({'LR': f"{current_lr:.1e}", 'Loss': f"{loss.item()*args.accumulation_steps:.3f}"})
            
        # --- ВАЛИДАЦИЯ ---
        model.eval()
        val_loss, val_labels, val_preds, val_probs = 0.0, [], [], []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="[Valid]"):
                inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    logits = model(inputs, return_embeddings=False)
                    loss = criterion(logits, labels)
                
                val_loss += loss.item()
                probs = torch.softmax(logits.float(), dim=1)
                val_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        scheduler.step()
        
        # Сбор метрик эпохи
        v_loss = val_loss / len(val_loader)
        v_acc = accuracy_score(val_labels, val_preds)
        v_bacc = balanced_accuracy_score(val_labels, val_preds)
        v_mcc = matthews_corrcoef(val_labels, val_preds)
        t_bacc = balanced_accuracy_score(train_labels, train_preds)
        
        tracker.update(epoch, train_loss/len(train_loader), v_loss, accuracy_score(train_labels, train_preds), v_acc, t_bacc, v_bacc)
        
        print(f"\n[SUMMARY] LR: {current_lr:.2e} | Loss: {v_loss:.4f} | Acc: {v_acc:.4f} | B-Acc: {v_bacc:.4f} | MCC: {v_mcc:.4f}\n")
        
        early_stopping(v_bacc, model)
        if early_stopping.early_stop:
            print("[!] Early Stopping triggered.")
            break

    # --- ФИНАЛЬНЫЙ ОТЧЕТ ПОСЛЕ ОБУЧЕНИЯ ---
    print("\n[*] Генерация финального медицинского отчета...")
    model.load_state_dict(torch.load(exp_dir / 'best_model.pth', weights_only=True))
    model.eval()
    
    test_preds, test_labels, test_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, memory_format=torch.channels_last)
            with torch.amp.autocast('cuda'):
                logits = model(inputs, return_embeddings=False)
            probs = torch.softmax(logits.float(), dim=1)
            test_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            test_labels.extend(labels.numpy())
            test_probs.extend(probs.cpu().numpy())
            
    adv_metrics = calculate_advanced_metrics(test_labels, test_preds, np.array(test_probs))
    save_medical_report(exp_dir, test_labels, test_preds, np.array(test_probs), adv_metrics)
    plot_training_results(tracker, test_labels, test_preds, save_path=str(exp_dir / 'training_results.png'))
    
    print(f"[SUCCESS] Результаты сохранены в: {exp_dir}")

if __name__ == '__main__':
    main()