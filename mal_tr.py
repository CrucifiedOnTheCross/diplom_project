import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from model import JointSkinLesionClassifier

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
MALIGNANT_CLASSES = {'akiec', 'bcc', 'mel'}
MALIGNANT_IDXS = [CLASS_NAMES.index(c) for c in MALIGNANT_CLASSES]


def clean_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            cleaned[k.replace("_orig_mod.", "")] = v
        else:
            cleaned[k] = v
    return cleaned


def load_model(model_path: str, device: str = "cuda"):
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    model = JointSkinLesionClassifier(num_classes=len(CLASS_NAMES)).to(device_obj)

    state = torch.load(model_path, map_location=device_obj)
    state = clean_state_dict(state)
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, device_obj


def build_loader(data_dir: str, batch_size: int = 32):
    transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    print("Классы в датасете:", dataset.classes)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return dataset, loader


@torch.no_grad()
def collect_binary_scores(model, loader, device):
    y_true_bin = []
    p_malignant = []
    argmax_bin = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images, return_embeddings=False)
        probs = torch.softmax(logits, dim=1)

        malignant_prob = probs[:, MALIGNANT_IDXS].sum(dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true_bin.extend([1 if CLASS_NAMES[int(y)] in MALIGNANT_CLASSES else 0 for y in labels.cpu().numpy()])
        p_malignant.extend(malignant_prob.cpu().numpy().tolist())
        argmax_bin.extend([1 if CLASS_NAMES[int(p)] in MALIGNANT_CLASSES else 0 for p in preds.cpu().numpy()])

    return np.array(y_true_bin), np.array(p_malignant), np.array(argmax_bin)


def evaluate_thresholds(y_true_bin, p_malignant):
    thresholds = np.linspace(0.0, 1.0, 201)

    rows = []
    for thr in thresholds:
        y_pred_bin = (p_malignant >= thr).astype(int)

        precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        acc = accuracy_score(y_true_bin, y_pred_bin)
        bacc = balanced_accuracy_score(y_true_bin, y_pred_bin)
        mcc = matthews_corrcoef(y_true_bin, y_pred_bin) if len(np.unique(y_pred_bin)) > 1 else 0.0

        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden = recall + specificity - 1.0

        rows.append({
            "threshold": thr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "mcc": mcc,
            "youden": youden,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

    return rows


def save_threshold_csv(rows, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold", "precision", "recall", "f1", "specificity",
                "accuracy", "balanced_accuracy", "mcc", "youden",
                "tp", "fp", "fn", "tn"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_threshold_curves(rows, out_png):
    thresholds = [r["threshold"] for r in rows]
    precision = [r["precision"] for r in rows]
    recall = [r["recall"] for r in rows]
    f1 = [r["f1"] for r in rows]
    specificity = [r["specificity"] for r in rows]
    bacc = [r["balanced_accuracy"] for r in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision, label="Precision")
    plt.plot(thresholds, recall, label="Recall")
    plt.plot(thresholds, f1, label="F1")
    plt.plot(thresholds, specificity, label="Specificity")
    plt.plot(thresholds, bacc, label="Balanced Accuracy")
    plt.xlabel("Threshold for malignant probability")
    plt.ylabel("Score")
    plt.title("Malignant vs Benign Threshold Analysis")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_pr_curve(y_true_bin, p_malignant, out_png):
    precision, recall, _ = precision_recall_curve(y_true_bin, p_malignant)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve: Malignant vs Benign")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    return pr_auc


def plot_roc_curve(y_true_bin, p_malignant, out_png):
    fpr, tpr, _ = roc_curve(y_true_bin, p_malignant)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Malignant vs Benign")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    return roc_auc


def evaluate_binary_predictions(y_true_bin, y_pred_bin, name="argmax"):
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    acc = accuracy_score(y_true_bin, y_pred_bin)
    bacc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    mcc = matthews_corrcoef(y_true_bin, y_pred_bin) if len(np.unique(y_pred_bin)) > 1 else 0.0
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n=== {name} ===")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1:                {f1:.4f}")
    print(f"Specificity:       {specificity:.4f}")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bacc:.4f}")
    print(f"MCC:               {mcc:.4f}")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    return {
        "name": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "mcc": mcc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def print_key_thresholds(rows):
    best_f1 = max(rows, key=lambda x: x["f1"])
    best_youden = max(rows, key=lambda x: x["youden"])

    print("\nЛучший threshold по F1:")
    print(
        f"threshold={best_f1['threshold']:.3f}, "
        f"precision={best_f1['precision']:.4f}, "
        f"recall={best_f1['recall']:.4f}, "
        f"f1={best_f1['f1']:.4f}, "
        f"specificity={best_f1['specificity']:.4f}, "
        f"bacc={best_f1['balanced_accuracy']:.4f}, "
        f"mcc={best_f1['mcc']:.4f}"
    )

    print("\nЛучший threshold по Youden index:")
    print(
        f"threshold={best_youden['threshold']:.3f}, "
        f"precision={best_youden['precision']:.4f}, "
        f"recall={best_youden['recall']:.4f}, "
        f"f1={best_youden['f1']:.4f}, "
        f"specificity={best_youden['specificity']:.4f}, "
        f"bacc={best_youden['balanced_accuracy']:.4f}, "
        f"mcc={best_youden['mcc']:.4f}"
    )

    target_recalls = [0.85, 0.90, 0.95]
    for target in target_recalls:
        candidates = [r for r in rows if r["recall"] >= target]
        if candidates:
            best = max(candidates, key=lambda x: x["precision"])
            print(f"\nЛучший threshold при recall >= {target:.2f}:")
            print(
                f"threshold={best['threshold']:.3f}, "
                f"precision={best['precision']:.4f}, "
                f"recall={best['recall']:.4f}, "
                f"f1={best['f1']:.4f}, "
                f"specificity={best['specificity']:.4f}, "
                f"bacc={best['balanced_accuracy']:.4f}, "
                f"mcc={best['mcc']:.4f}"
            )
        else:
            print(f"\nНе найден threshold с recall >= {target:.2f}")


def main():
    model_path = "science_folder/20_raw_supcon/best_model.pth"
    test_dir = "dataset/test"
    out_dir = Path("binary_threshold_analysis_20_raw_supcon")
    out_dir.mkdir(exist_ok=True)

    model, device = load_model(model_path, device="cuda")
    _, loader = build_loader(test_dir, batch_size=32)

    y_true_bin, p_malignant, argmax_bin = collect_binary_scores(model, loader, device)

    # Базовая бинарная оценка через argmax
    evaluate_binary_predictions(y_true_bin, argmax_bin, name="Argmax baseline")

    # Threshold analysis
    rows = evaluate_thresholds(y_true_bin, p_malignant)
    save_threshold_csv(rows, out_dir / "malignant_thresholds.csv")

    plot_threshold_curves(rows, out_dir / "malignant_threshold_curves.png")
    pr_auc = plot_pr_curve(y_true_bin, p_malignant, out_dir / "malignant_pr_curve.png")
    roc_auc = plot_roc_curve(y_true_bin, p_malignant, out_dir / "malignant_roc_curve.png")

    print(f"\nPR-AUC malignant = {pr_auc:.4f}")
    print(f"ROC-AUC malignant = {roc_auc:.4f}")

    print_key_thresholds(rows)

    # Сравнение argmax vs лучший threshold по F1
    best_f1_row = max(rows, key=lambda x: x["f1"])
    best_thr = best_f1_row["threshold"]
    y_pred_best = (p_malignant >= best_thr).astype(int)

    evaluate_binary_predictions(
        y_true_bin,
        y_pred_best,
        name=f"Thresholded malignant (best F1, thr={best_thr:.3f})"
    )

    print(f"\nФайлы сохранены в: {out_dir.resolve()}")


if __name__ == "__main__":
    main()