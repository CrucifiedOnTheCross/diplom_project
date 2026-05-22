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
)
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from model import JointSkinLesionClassifier

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
MEL_IDX = CLASS_NAMES.index("mel")


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


def build_test_loader(test_dir: str, batch_size: int = 32):
    transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(test_dir, transform=transform)

    # Проверка порядка классов
    print("Классы в датасете:", dataset.classes)
    if dataset.classes != CLASS_NAMES:
        print("ВНИМАНИЕ: порядок классов отличается от ожидаемого.")
        print("Ожидается:", CLASS_NAMES)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return dataset, loader


@torch.no_grad()
def collect_melanoma_scores(model, loader, device):
    y_true_bin = []
    mel_probs = []
    pred_classes = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images, return_embeddings=False)
        probs = torch.softmax(logits, dim=1)

        mel_prob = probs[:, MEL_IDX]
        preds = torch.argmax(probs, dim=1)

        y_true_bin.extend((labels == MEL_IDX).long().cpu().numpy().tolist())
        mel_probs.extend(mel_prob.cpu().numpy().tolist())
        pred_classes.extend(preds.cpu().numpy().tolist())

    return np.array(y_true_bin), np.array(mel_probs), np.array(pred_classes)


def evaluate_thresholds(y_true_bin, mel_probs):
    thresholds = np.linspace(0.0, 1.0, 201)

    rows = []
    for thr in thresholds:
        y_pred_bin = (mel_probs >= thr).astype(int)

        precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(
            y_true_bin, y_pred_bin, labels=[0, 1]
        ).ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        rows.append({
            "threshold": thr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
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
            fieldnames=["threshold", "precision", "recall", "f1", "specificity", "tp", "fp", "fn", "tn"]
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_threshold_curves(rows, out_png):
    thresholds = [r["threshold"] for r in rows]
    precision = [r["precision"] for r in rows]
    recall = [r["recall"] for r in rows]
    f1 = [r["f1"] for r in rows]
    specificity = [r["specificity"] for r in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision, label="Precision")
    plt.plot(thresholds, recall, label="Recall")
    plt.plot(thresholds, f1, label="F1")
    plt.plot(thresholds, specificity, label="Specificity")
    plt.xlabel("Threshold for melanoma probability")
    plt.ylabel("Score")
    plt.title("Melanoma Threshold Analysis")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_pr_curve(y_true_bin, mel_probs, out_png):
    precision, recall, _ = precision_recall_curve(y_true_bin, mel_probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for Melanoma")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    return pr_auc


def plot_roc_curve(y_true_bin, mel_probs, out_png):
    fpr, tpr, _ = roc_curve(y_true_bin, mel_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Melanoma")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    return roc_auc


def print_key_thresholds(rows):
    best_f1 = max(rows, key=lambda x: x["f1"])

    print("\nЛучший threshold по F1:")
    print(
        f"threshold={best_f1['threshold']:.3f}, "
        f"precision={best_f1['precision']:.4f}, "
        f"recall={best_f1['recall']:.4f}, "
        f"f1={best_f1['f1']:.4f}, "
        f"specificity={best_f1['specificity']:.4f}"
    )

    targets = [0.2, 0.3, 0.4, 0.5, 0.6]
    print("\nКонтрольные пороги:")
    for t in targets:
        nearest = min(rows, key=lambda x: abs(x["threshold"] - t))
        print(
            f"thr={nearest['threshold']:.2f} | "
            f"P={nearest['precision']:.4f} | "
            f"R={nearest['recall']:.4f} | "
            f"F1={nearest['f1']:.4f} | "
            f"Spec={nearest['specificity']:.4f}"
        )

    # Порог с recall >= 0.80 и максимальной precision
    high_recall = [r for r in rows if r["recall"] >= 0.80]
    if high_recall:
        best_high_recall = max(high_recall, key=lambda x: x["precision"])
        print("\nЛучший порог при recall >= 0.80:")
        print(
            f"threshold={best_high_recall['threshold']:.3f}, "
            f"precision={best_high_recall['precision']:.4f}, "
            f"recall={best_high_recall['recall']:.4f}, "
            f"f1={best_high_recall['f1']:.4f}, "
            f"specificity={best_high_recall['specificity']:.4f}"
        )
    else:
        print("\nНе найден порог с recall >= 0.80")


def main():
    model_path = "science_folder/20_raw_supcon/best_model.pth"
    test_dir = "dataset/test"
    out_dir = Path("threshold_analysis_20_raw_supcon")
    out_dir.mkdir(exist_ok=True)

    model, device = load_model(model_path, device="cuda")
    _, loader = build_test_loader(test_dir, batch_size=32)

    y_true_bin, mel_probs, pred_classes = collect_melanoma_scores(model, loader, device)

    rows = evaluate_thresholds(y_true_bin, mel_probs)
    save_threshold_csv(rows, out_dir / "melanoma_thresholds.csv")

    plot_threshold_curves(rows, out_dir / "melanoma_threshold_curves.png")
    pr_auc = plot_pr_curve(y_true_bin, mel_probs, out_dir / "melanoma_pr_curve.png")
    roc_auc = plot_roc_curve(y_true_bin, mel_probs, out_dir / "melanoma_roc_curve.png")

    print(f"\nPR-AUC melanoma = {pr_auc:.4f}")
    print(f"ROC-AUC melanoma = {roc_auc:.4f}")

    print_key_thresholds(rows)

    print(f"\nФайлы сохранены в: {out_dir.resolve()}")


if __name__ == "__main__":
    main()