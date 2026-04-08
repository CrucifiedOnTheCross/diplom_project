import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    matthews_corrcoef, 
    average_precision_score, 
    confusion_matrix, 
    classification_report, 
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

# Стандартный порядок классов для HAM10000
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
# Индексы злокачественных новообразований: akiec, bcc, mel
MALIGNANT_IDX = [0, 1, 4]

import csv

class HistoryTracker:
    """
    Класс для хранения истории метрик и сохранения их на диск 'на лету' (Streaming).
    Защищает от потери данных при падении скрипта.
    """
    def __init__(self, exp_dir=None):
        self.history = {
            'train_loss': [], 'val_loss': [], 
            'train_acc': [], 'val_acc': [], 
            'train_bacc': [], 'val_bacc': []
        }
        self.csv_path = exp_dir / 'training_history.csv' if exp_dir else None
        
        # Создаем файл и пишем заголовки, если передан путь
        if self.csv_path:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Train_Acc', 'Val_Acc', 'Train_BAcc', 'Val_BAcc'])

    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, train_bacc, val_bacc):
        # 1. Обновляем оперативную память (для графиков)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['train_bacc'].append(train_bacc)
        self.history['val_bacc'].append(val_bacc)
        
        # 2. Дописываем строку в CSV на диск (Streaming)
        if self.csv_path:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([
                    epoch, 
                    f"{train_loss:.4f}", f"{val_loss:.4f}", 
                    f"{train_acc:.4f}", f"{val_acc:.4f}", 
                    f"{train_bacc:.4f}", f"{val_bacc:.4f}"
                ])

class EarlyStopping:
    """Класс для остановки обучения при стагнации целевой метрики."""
    def __init__(self, patience=10, mode='max', min_delta=1e-4, save_path='best_model.pth'):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_metric, model):
        score = current_metric if self.mode == 'max' else -current_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)

def calculate_advanced_metrics(y_true, y_pred, y_probs):
    """Расчет расширенного набора медицинских метрик."""
    # Подготовка данных для многоклассового анализа
    y_true_bin = label_binarize(y_true, classes=range(len(CLASS_NAMES)))
    
    # 1. PR-AUC (Macro) и MCC
    pr_auc = average_precision_score(y_true_bin, y_probs, average="macro")
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # 2. ROC-AUC
    macro_roc_auc = roc_auc_score(y_true_bin, y_probs, average="macro", multi_class="ovr")
    weighted_roc_auc = roc_auc_score(y_true_bin, y_probs, average="weighted", multi_class="ovr")
    
    # 3. Бинарные метрики (Злокачественные vs Доброкачественные)
    y_true_binarized = np.isin(y_true, MALIGNANT_IDX).astype(int)
    y_pred_binarized = np.isin(y_pred, MALIGNANT_IDX).astype(int)
    
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true_binarized, y_pred_binarized).ravel()
    bin_sensitivity = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
    bin_specificity = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0

    # 4. Специфичность для каждого отдельного класса
    cm = confusion_matrix(y_true, y_pred)
    specificities = {}
    for i, cls_name in enumerate(CLASS_NAMES):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificities[cls_name] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'PR-AUC': pr_auc,
        'MCC': mcc,
        'Macro ROC-AUC': macro_roc_auc,
        'Weighted ROC-AUC': weighted_roc_auc,
        'Bin Sensitivity': bin_sensitivity,
        'Bin Specificity': bin_specificity,
        'Per-Class Specificity': specificities,
        'CM_Norm': confusion_matrix(y_true, y_pred, normalize='true')
    }

def plot_training_results(history_tracker, y_true_val, y_pred_val, save_path='training_results.png'):
    """Визуализация графиков обучения и нормализованной матрицы ошибок."""
    h = history_tracker.history
    epochs = range(1, len(h['train_loss']) + 1)
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # График Loss
    axs[0, 0].plot(epochs, h['train_loss'], label='Train Loss')
    axs[0, 0].plot(epochs, h['val_loss'], label='Val Loss')
    axs[0, 0].set_title('Loss History')
    axs[0, 0].legend(); axs[0, 0].grid(True)
    
    # График Accuracy
    axs[0, 1].plot(epochs, h['train_acc'], label='Train Acc')
    axs[0, 1].plot(epochs, h['val_acc'], label='Val Acc')
    axs[0, 1].set_title('Accuracy History')
    axs[0, 1].legend(); axs[0, 1].grid(True)
    
    # График Balanced Accuracy
    axs[1, 0].plot(epochs, h['train_bacc'], label='Train Balanced Acc')
    axs[1, 0].plot(epochs, h['val_bacc'], label='Val Balanced Acc')
    axs[1, 0].set_title('Balanced Accuracy History')
    axs[1, 0].legend(); axs[1, 0].grid(True)
    
    # Нормализованная Матрица Ошибок
    cm_norm = confusion_matrix(y_true_val, y_pred_val, normalize='true')
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axs[1, 1], 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, vmin=0, vmax=1)
    axs[1, 1].set_title('Normalized Confusion Matrix')
    axs[1, 1].set_xlabel('Predicted'); axs[1, 1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def save_medical_report(exp_dir, y_true, y_pred, y_probs, adv_metrics):
    """Генерация подробного текстового отчета по результатам эксперимента."""
    report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    cm_norm = adv_metrics['CM_Norm']
    
    with open(exp_dir / 'medical_metrics_report.txt', 'w', encoding='utf-8') as f:
        f.write("====================================================\n")
        f.write("      MEDICAL CLASSIFICATION METRICS REPORT\n")
        f.write("====================================================\n\n")
        
        f.write("1. PRIMARY CLASSIFICATION METRICS (Precision, Recall/TPR, F1)\n")
        f.write("----------------------------------------------------\n")
        f.write(report_text + "\n\n")
        
        f.write("2. ADVANCED MEDICAL METRICS\n")
        f.write("----------------------------------------------------\n")
        f.write(f"Standard Accuracy:          {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"Balanced Accuracy:          {balanced_accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"MCC (Matthews Corr):        {adv_metrics['MCC']:.4f}\n")
        f.write(f"PR-AUC (Macro):             {adv_metrics['PR-AUC']:.4f}\n")
        f.write(f"Macro Average ROC-AUC:      {adv_metrics['Macro ROC-AUC']:.4f}\n")
        f.write(f"Weighted Average ROC-AUC:   {adv_metrics['Weighted ROC-AUC']:.4f}\n\n")
        
        f.write("Binary Evaluation (Malignant vs Benign):\n")
        f.write(f" - Sensitivity (Recall):    {adv_metrics['Bin Sensitivity']:.4f}\n")
        f.write(f" - Specificity:             {adv_metrics['Bin Specificity']:.4f}\n\n")

        f.write("Per-Class Specificity:\n")
        for cls_name, spec in adv_metrics['Per-Class Specificity'].items():
            f.write(f" - {cls_name:<10}: {spec:.4f}\n")
            
        f.write("\n3. MISCLASSIFICATION ANALYSIS (Ошибки > 5%)\n")
        f.write("----------------------------------------------------\n")
        for i, true_cls in enumerate(CLASS_NAMES):
            for j, pred_cls in enumerate(CLASS_NAMES):
                if i != j and cm_norm[i, j] > 0.05:
                    f.write(f" - ОШИБКА: '{true_cls}' принят за '{pred_cls}' в {cm_norm[i, j]*100:.1f}% случаев.\n")