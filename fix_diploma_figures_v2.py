#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Исправляет проблемные рисунки после build_all_diploma_figures.py.

Что делает:
1. Рис. 2.4 / 2.5: строит реальные примеры preprocessing: original -> mask -> result.
2. Рис. 3.4: строит дерево каталогов dataset/train/valid/test.
3. Рис. 3.1, 3.2, 3.5, 3.8: заменяет примитивные flowchart на многоуровневые схемы.
4. Рис. 3.13 / 4.23: строит отдельную normalized confusion matrix 7x7.
5. Рис. 3.16: строит reliability diagram before/after.
6. Рис. 4.16: строит ECE before/after bar chart.
7. Рис. 3.17 / 4.17: строит threshold curves malignant/benign.
8. Рис. 3.18 / 4.18: строит threshold curves melanoma/rest.
9. Рис. 4.7, 4.9, 4.13: фильтрует только нужные группы экспериментов.
10. Рис. 4.14: делает читаемый horizontal bar chart для GAN-mix.
11. Рис. 4.15: строит scatter FID vs Balanced Accuracy.
12. Рис. 4.22: строит dot plot вариативности по seed.
13. По флагу --archive пакует diploma_figures в архив.

Запуск:
  python fix_diploma_figures_v2.py

С архивом:
  python fix_diploma_figures_v2.py --archive --archive-name diploma_figures_fixed.tar.gz
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


# =====================================================================
# Общие утилиты
# =====================================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path, dpi: int = 300):
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def read_csv_auto(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) > 1:
                return df
        except Exception:
            pass

    return None


def to_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def placeholder(out_path: Path, title: str, required: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    ax.text(0.5, 0.65, title, ha="center", va="center",
            fontsize=17, fontweight="bold", wrap=True)
    ax.text(0.5, 0.42, "Не удалось построить реальный график.",
            ha="center", va="center", fontsize=13)
    ax.text(0.5, 0.25, f"Ожидаемые данные: {required}",
            ha="center", va="center", fontsize=11, wrap=True)
    save_fig(out_path)


def normalize_exp_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None

    numeric_cols = [
        "Accuracy", "Balanced_Accuracy", "MCC", "F1_Macro", "F1_Weighted",
        "PR_AUC", "ROC_AUC_Macro", "ROC_AUC_Weighted",
    ]
    for cls in CLASS_NAMES:
        numeric_cols += [
            f"{cls}_Recall", f"{cls}_Specificity", f"{cls}_Precision", f"{cls}_F1"
        ]

    return to_num(df.copy(), [c for c in numeric_cols if c in df.columns])


# =====================================================================
# Базовая графика
# =====================================================================

def draw_box(ax, xy, w, h, text, fontsize=9, lw=1.4):
    x, y = xy
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=lw,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, wrap=True)


def draw_arrow(ax, start, end, lw=1.3):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=lw),
    )


def multilevel_pipeline(out_path: Path, title: str, lanes: Dict[str, List[str]]):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, title, ha="center", va="center",
            fontsize=16, fontweight="bold")

    lane_names = list(lanes.keys())
    y_positions = np.linspace(0.78, 0.18, len(lane_names))

    for lane_idx, lane in enumerate(lane_names):
        y = y_positions[lane_idx]
        ax.text(0.03, y + 0.045, lane, ha="left", va="center",
                fontsize=11, fontweight="bold")

        boxes = lanes[lane]
        n = len(boxes)
        x_positions = np.linspace(0.18, 0.83, n)

        for i, txt in enumerate(boxes):
            draw_box(ax, (x_positions[i], y), 0.13, 0.09, txt, fontsize=8)
            if i < n - 1:
                draw_arrow(
                    ax,
                    (x_positions[i] + 0.13, y + 0.045),
                    (x_positions[i + 1], y + 0.045),
                )

    # межуровневые стрелки
    if len(y_positions) >= 2:
        for i in range(len(y_positions) - 1):
            draw_arrow(ax, (0.50, y_positions[i]), (0.50, y_positions[i + 1] + 0.09))

    save_fig(out_path)


# =====================================================================
# 2.4 / 2.5: реальные preprocessing examples
# =====================================================================

def find_example_image(dataset_root: Path) -> Optional[Path]:
    candidates = []
    for split in ["train", "valid", "test"]:
        for cls in ["mel", "bcc", "akiec", "bkl", "nv", "df", "vasc"]:
            d = dataset_root / split / cls
            if d.exists():
                candidates += sorted(d.glob("*.jpg")) + sorted(d.glob("*.jpeg")) + sorted(d.glob("*.png"))
    return candidates[0] if candidates else None


def build_preprocessing_figures(dataset_root: Path, ch2_dir: Path):
    try:
        import cv2
        from PIL import Image
    except Exception:
        placeholder(
            ch2_dir / "fig_2_4_preprocessing_v1.png",
            "Первый вариант предобработки",
            "opencv-python, pillow"
        )
        placeholder(
            ch2_dir / "fig_2_5_preprocessing_v2.png",
            "Второй вариант предобработки",
            "opencv-python, pillow"
        )
        return

    img_path = find_example_image(dataset_root)
    if img_path is None:
        placeholder(
            ch2_dir / "fig_2_4_preprocessing_v1.png",
            "Первый вариант предобработки",
            "dataset/train|valid|test/<class>/*.jpg"
        )
        placeholder(
            ch2_dir / "fig_2_5_preprocessing_v2.png",
            "Второй вариант предобработки",
            "dataset/train|valid|test/<class>/*.jpg"
        )
        return

    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return

    img_bgr = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # V1: gray-world + fixed blackhat + inpainting
    img_v1 = img_rgb.copy().astype(np.float32)
    means = img_v1.reshape(-1, 3).mean(axis=0)
    avg = means.mean()
    img_v1 = np.clip(img_v1 * (avg / (means + 1e-6)), 0, 255).astype(np.uint8)

    gray_v1 = cv2.cvtColor(img_v1, cv2.COLOR_RGB2GRAY)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat1 = cv2.morphologyEx(gray_v1, cv2.MORPH_BLACKHAT, kernel1)
    _, mask1 = cv2.threshold(blackhat1, 15, 255, cv2.THRESH_BINARY)
    result1 = cv2.inpaint(img_v1, mask1, 3, cv2.INPAINT_TELEA)

    # V2: adaptive blackhat + Otsu + inpainting + CLAHE
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat2 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel2)
    blackhat2 = cv2.GaussianBlur(blackhat2, (3, 3), 0)
    _, mask2 = cv2.threshold(blackhat2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask2 = cv2.medianBlur(mask2, 3)
    mask2 = cv2.dilate(mask2, np.ones((3, 3), np.uint8), iterations=1)
    inpainted = cv2.inpaint(img_bgr, mask2, 3, cv2.INPAINT_TELEA)

    lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    result2_bgr = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    result2 = cv2.cvtColor(result2_bgr, cv2.COLOR_BGR2RGB)

    # Figure 2.4
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    items = [
        (img_rgb, "Исходное изображение"),
        (img_v1, "Нормализация цвета"),
        (mask1, "Маска волос"),
        (result1, "Inpainting / результат"),
    ]
    for ax, (img, title) in zip(axes, items):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Рисунок 2.4 — Первый вариант предобработки", fontsize=14, fontweight="bold")
    save_fig(ch2_dir / "fig_2_4_preprocessing_v1.png")

    # Figure 2.5
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    items = [
        (img_rgb, "Исходное изображение"),
        (mask2, "Адаптивная маска волос"),
        (result1, "Вариант 1"),
        (result2, "Вариант 2: inpainting + CLAHE"),
    ]
    for ax, (img, title) in zip(axes, items):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Рисунок 2.5 — Второй вариант предобработки", fontsize=14, fontweight="bold")
    save_fig(ch2_dir / "fig_2_5_preprocessing_v2.png")


# =====================================================================
# 3.4: дерево каталогов
# =====================================================================

def draw_dataset_tree(out_path: Path):
    tree_text = """dataset/
├── train/
│   ├── akiec/
│   ├── bcc/
│   ├── bkl/
│   ├── df/
│   ├── mel/
│   ├── nv/
│   └── vasc/
├── valid/
│   ├── akiec/
│   ├── bcc/
│   ├── bkl/
│   ├── df/
│   ├── mel/
│   ├── nv/
│   └── vasc/
└── test/
    ├── akiec/
    ├── bcc/
    ├── bkl/
    ├── df/
    ├── mel/
    ├── nv/
    └── vasc/"""

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.axis("off")
    ax.text(0.5, 0.95, "Файловая структура подготовленного датасета",
            ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.08, 0.86, tree_text, ha="left", va="top",
            family="monospace", fontsize=13)
    ax.text(
        0.5, 0.06,
        "Структура формируется после lesion_id split; valid/test остаются real-only.",
        ha="center", va="center", fontsize=11,
    )
    save_fig(out_path)


# =====================================================================
# Confusion matrix из модели
# =====================================================================

def load_joint_model(model_path: Path, device: str):
    import torch
    from model import JointSkinLesionClassifier

    model = JointSkinLesionClassifier(num_classes=len(CLASS_NAMES)).to(device)
    state = torch.load(model_path, map_location=device)

    cleaned = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            cleaned[k.replace("_orig_mod.", "")] = v
        else:
            cleaned[k] = v

    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def compute_predictions(model_path: Path, data_dir: Path, device_str: str = "cuda"):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import v2

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    ds = datasets.ImageFolder(str(data_dir), transform=transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = load_joint_model(model_path, device)

    y_true, y_pred, probs_all = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x, return_embeddings=False)
            probs = torch.softmax(logits.float(), dim=1)
            pred = torch.argmax(probs, dim=1)

            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            probs_all.extend(probs.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(probs_all), ds.classes


def plot_confusion_matrix_from_model(
    model_path: Path,
    data_dir: Path,
    out_paths: List[Path],
):
    try:
        from sklearn.metrics import confusion_matrix
    except Exception:
        for out in out_paths:
            placeholder(out, "Нормализованная матрица ошибок", "scikit-learn")
        return

    if not model_path.exists() or not data_dir.exists():
        for out in out_paths:
            placeholder(out, "Нормализованная матрица ошибок", f"{model_path}, {data_dir}")
        return

    try:
        y_true, y_pred, _, classes = compute_predictions(model_path, data_dir)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))), normalize="true")
    except Exception as e:
        for out in out_paths:
            placeholder(out, "Нормализованная матрица ошибок", f"ошибка инференса: {e}")
        return

    for out in out_paths:
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("Нормализованная матрица ошибок выбранной модели")

        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        save_fig(out)


# =====================================================================
# Reliability diagram and ECE
# =====================================================================

def find_temperature(report_path: Path) -> float:
    if not report_path.exists():
        return 1.0

    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return 1.0

    candidates = [
        "temperature", "Temperature", "T", "optimal_temperature",
        "best_temperature",
    ]

    def search(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in candidates:
                    try:
                        return float(v)
                    except Exception:
                        pass
                nested = search(v)
                if nested is not None:
                    return nested
        return None

    val = search(data)
    return val if val is not None and val > 0 else 1.0


def reliability_bins(y_true, probs, n_bins=10):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if mask.sum() == 0:
            centers.append((lo + hi) / 2)
            accs.append(np.nan)
            confs.append(np.nan)
            counts.append(0)
        else:
            centers.append((lo + hi) / 2)
            accs.append(correct[mask].mean())
            confs.append(conf[mask].mean())
            counts.append(mask.sum())

    return np.array(centers), np.array(accs), np.array(confs), np.array(counts)


def ece_score(y_true, probs, n_bins=10):
    centers, accs, confs, counts = reliability_bins(y_true, probs, n_bins)
    total = counts.sum()
    if total == 0:
        return np.nan

    ece = 0.0
    for a, c, n in zip(accs, confs, counts):
        if n > 0:
            ece += (n / total) * abs(a - c)
    return ece


def plot_reliability_from_model(
    model_path: Path,
    data_dir: Path,
    calibration_report: Path,
    out_path: Path,
):
    if not model_path.exists() or not data_dir.exists():
        placeholder(out_path, "Диаграмма надежности до и после калибровки", f"{model_path}, {data_dir}")
        return

    try:
        y_true, _, probs_before, _ = compute_predictions(model_path, data_dir)
    except Exception as e:
        placeholder(out_path, "Диаграмма надежности до и после калибровки", f"ошибка инференса: {e}")
        return

    temperature = find_temperature(calibration_report)

    logits_approx = np.log(np.clip(probs_before, 1e-12, 1.0))
    logits_scaled = logits_approx / temperature
    exps = np.exp(logits_scaled - logits_scaled.max(axis=1, keepdims=True))
    probs_after = exps / exps.sum(axis=1, keepdims=True)

    ece_before = ece_score(y_true, probs_before)
    ece_after = ece_score(y_true, probs_after)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, probs, title, ece in [
        (axes[0], probs_before, "Before calibration", ece_before),
        (axes[1], probs_after, f"After temperature scaling (T={temperature:.3f})", ece_after),
    ]:
        centers, accs, confs, counts = reliability_bins(y_true, probs)
        ax.bar(centers, np.nan_to_num(accs), width=0.08, alpha=0.7, label="Empirical accuracy")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title(f"{title}\nECE={ece:.4f}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    save_fig(out_path)


def collect_ece_from_reports(science_dir: Path) -> pd.DataFrame:
    rows = []

    def find_key(obj, names):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() in names:
                    try:
                        return float(v)
                    except Exception:
                        pass
                nested = find_key(v, names)
                if nested is not None:
                    return nested
        return None

    for exp_dir in sorted(science_dir.glob("*")):
        if not exp_dir.is_dir():
            continue

        report = exp_dir / "calibration_report.json"
        if not report.exists():
            continue

        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except Exception:
            continue

        before = find_key(data, {"ece_before", "before_ece", "ece_uncalibrated", "uncalibrated_ece"})
        after = find_key(data, {"ece_after", "after_ece", "ece_calibrated", "calibrated_ece", "ece"})

        if before is not None or after is not None:
            rows.append({
                "Experiment_Folder": exp_dir.name,
                "ECE_before": before,
                "ECE_after": after,
            })

    return pd.DataFrame(rows)


def plot_ece_grouped(science_dir: Path, out_path: Path):
    df = collect_ece_from_reports(science_dir)

    selected = [
        "01_base_raw",
        "20_raw_supcon",
        "24_raw_supcon_weighted",
        "27_gan_focal_g1",
        "baseline_real_only_raw_ganmix",
        "gan_bcc_25_raw_ganmix",
    ]

    if df.empty:
        placeholder(out_path, "Изменение ECE до и после температурной калибровки", "science_folder/*/calibration_report.json")
        return

    sub = df[df["Experiment_Folder"].isin(selected)].copy()
    if sub.empty:
        sub = df.head(8).copy()

    x = np.arange(len(sub))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, sub["ECE_before"], width, label="ECE before")
    ax.bar(x + width / 2, sub["ECE_after"], width, label="ECE after")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["Experiment_Folder"], rotation=35, ha="right")
    ax.set_ylabel("ECE ↓")
    ax.set_title("Изменение ECE до и после температурной калибровки")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    save_fig(out_path)


# =====================================================================
# Threshold curves
# =====================================================================

def standardize_threshold_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит разные варианты названий колонок threshold-таблиц к единому виду.
    Важно: после rename могут возникнуть дубли колонок, например recall и sensitivity
    одновременно превращаются в sensitivity. Поэтому дубли схлопываются.
    """
    df = df.copy()

    rename = {}
    for col in df.columns:
        lc = str(col).strip().lower()

        if lc in {"threshold", "thr", "t"}:
            rename[col] = "threshold"
        elif lc in {"recall", "sensitivity", "tpr"}:
            rename[col] = "sensitivity"
        elif lc in {"specificity", "tnr"}:
            rename[col] = "specificity"
        elif lc in {"precision", "ppv"}:
            rename[col] = "precision"
        elif lc in {"f1", "f1_score"}:
            rename[col] = "f1"
        elif lc in {"balanced_accuracy", "bacc", "balanced_acc"}:
            rename[col] = "balanced_accuracy"
        elif lc in {"mcc", "matthews"}:
            rename[col] = "mcc"
        elif lc in {"youden", "youden_index"}:
            rename[col] = "youden"

    df = df.rename(columns=rename)

    # Схлопываем дублирующиеся колонки после rename.
    # Если есть несколько колонок с одним именем, берём первую непустую по строке.
    if df.columns.duplicated().any():
        collapsed = {}
        for col in dict.fromkeys(df.columns):
            same = df.loc[:, df.columns == col]
            if same.shape[1] == 1:
                collapsed[col] = same.iloc[:, 0]
            else:
                collapsed[col] = same.bfill(axis=1).iloc[:, 0]
        df = pd.DataFrame(collapsed)

    numeric_cols = [
        "threshold",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "balanced_accuracy",
        "mcc",
        "youden",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def find_threshold_file(science_dir: Path, mode: str, exp_candidates: List[str]) -> Optional[Path]:
    names = [
        f"threshold_curve_after_{mode}.csv",
        f"threshold_curve_before_{mode}.csv",
        f"{mode}_thresholds.csv",
    ]

    for exp in exp_candidates:
        exp_dir = science_dir / exp
        for n in names:
            p = exp_dir / n
            if p.exists():
                return p

    for p in science_dir.glob(f"*/threshold_curve_after_{mode}.csv"):
        return p
    for p in science_dir.glob(f"*/threshold_curve_before_{mode}.csv"):
        return p

    return None


def plot_threshold_curves(science_dir: Path, mode: str, out_paths: List[Path]):
    title = "Пороговый анализ malignant/benign" if mode == "malignant" else "Пороговый анализ melanoma/rest"

    p = find_threshold_file(
        science_dir,
        mode,
        ["20_raw_supcon", "01_base_raw", "24_raw_supcon_weighted", "gan_bcc_25_raw_ganmix"]
    )

    if p is None:
        for out in out_paths:
            placeholder(out, title, f"science_folder/*/threshold_curve_after_{mode}.csv")
        return

    df = read_csv_auto(p)
    if df is None:
        for out in out_paths:
            placeholder(out, title, str(p))
        return

    df = standardize_threshold_df(df)
    if "threshold" not in df.columns:
        for out in out_paths:
            placeholder(out, title, "column threshold")
        return

    metrics = ["sensitivity", "specificity", "precision", "f1", "balanced_accuracy", "mcc", "youden"]
    metrics = [m for m in metrics if m in df.columns]

    for out in out_paths:
        fig, ax = plt.subplots(figsize=(10, 6))
        for m in metrics:
            ax.plot(df["threshold"], df[m], label=m)

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Metric value")
        ax.set_title(f"{title}\nИсточник: {p.parent.name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        save_fig(out)


# =====================================================================
# Filtered plots: 4.7, 4.9, 4.13, 4.14, 4.15, 4.22
# =====================================================================

def scatter_filtered(
    df: Optional[pd.DataFrame],
    exp_names: List[str],
    x_col: str,
    y_col: str,
    out_path: Path,
    title: str,
):
    if df is None or "Experiment_Folder" not in df.columns:
        placeholder(out_path, title, "all_experiments_summary.csv")
        return

    df = normalize_exp_df(df)
    sub = df[df["Experiment_Folder"].isin(exp_names)].copy()

    if sub.empty or x_col not in sub.columns or y_col not in sub.columns:
        placeholder(out_path, title, f"{exp_names}, {x_col}, {y_col}")
        return

    sub = sub.dropna(subset=[x_col, y_col])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(sub[x_col], sub[y_col], s=90)

    for _, r in sub.iterrows():
        ax.annotate(str(r["Experiment_Folder"]), (r[x_col], r[y_col]), fontsize=8)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    save_fig(out_path)


def plot_ganmix_horizontal(df: Optional[pd.DataFrame], out_path: Path):
    title = "Сравнение одноклассовых и комбинированных GAN-mix конфигураций"

    if df is None or "Experiment_Folder" not in df.columns:
        placeholder(out_path, title, "all_experiments_summary.csv")
        return

    df = normalize_exp_df(df)
    mask = df["Experiment_Folder"].astype(str).str.contains("raw_ganmix|baseline_real_only", regex=True)
    sub = df[mask].copy()

    if sub.empty:
        placeholder(out_path, title, "experiments containing raw_ganmix")
        return

    sub = sub.sort_values("MCC", ascending=True)

    y = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(10, max(5, len(sub) * 0.42)))

    ax.barh(y - 0.18, sub["Balanced_Accuracy"], height=0.35, label="Balanced Accuracy")
    ax.barh(y + 0.18, sub["MCC"], height=0.35, label="MCC")

    ax.set_yticks(y)
    ax.set_yticklabels(sub["Experiment_Folder"], fontsize=8)
    ax.set_xlabel("Metric value")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()

    save_fig(out_path)


def plot_fid_vs_bacc(df: Optional[pd.DataFrame], gan_best: Optional[pd.DataFrame], out_path: Path):
    title = "Сопоставление FID и Balanced Accuracy для GAN-mix экспериментов"

    if df is None or gan_best is None:
        placeholder(out_path, title, "all_experiments_summary.csv + gan_analysis/gan_best_snapshots.csv")
        return

    df = normalize_exp_df(df)
    gan_best = gan_best.copy()

    if "class_name" not in gan_best.columns or "best_fid" not in gan_best.columns:
        placeholder(out_path, title, "gan_best_snapshots.csv columns: class_name, best_fid")
        return

    gan_best["best_fid"] = pd.to_numeric(gan_best["best_fid"], errors="coerce")
    fid_map = dict(zip(gan_best["class_name"].astype(str), gan_best["best_fid"]))

    rows = []
    for cls in ["bcc", "akiec", "vasc", "df"]:
        pattern = f"gan_{cls}_"
        sub = df[df["Experiment_Folder"].astype(str).str.contains(pattern, regex=False)]
        for _, r in sub.iterrows():
            if cls in fid_map and not pd.isna(r.get("Balanced_Accuracy", np.nan)):
                rows.append({
                    "experiment": r["Experiment_Folder"],
                    "class": cls,
                    "fid": fid_map[cls],
                    "bacc": r["Balanced_Accuracy"],
                })

    if not rows:
        placeholder(out_path, title, "single-class GAN-mix experiments")
        return

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    for cls, g in plot_df.groupby("class"):
        ax.scatter(g["fid"], g["bacc"], s=90, label=cls)
        for _, r in g.iterrows():
            ax.annotate(str(r["experiment"]).replace("_raw_ganmix", ""), (r["fid"], r["bacc"]), fontsize=8)

    ax.set_xlabel("Best FID of class-specific GAN ↓")
    ax.set_ylabel("Balanced Accuracy of GAN-mix classifier ↑")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_fig(out_path)


def group_seed_exp(name: str) -> Optional[str]:
    if name.startswith("20_raw_supcon_seed") or name == "20_raw_supcon":
        return "20_raw_supcon"
    if name.startswith("13_gan_supcon_seed") or name == "13_gan_supcon":
        return "13_gan_supcon"
    if name.startswith("ablation_raw_focal_g1_weighted_seed") or name == "ablation_raw_focal_g1_weighted":
        return "ablation_raw_focal_g1_weighted"
    return None


def plot_seed_variability(df: Optional[pd.DataFrame], out_path: Path):
    title = "Вариативность метрик по seed"

    if df is None or "Experiment_Folder" not in df.columns:
        placeholder(out_path, title, "all_experiments_summary.csv")
        return

    df = normalize_exp_df(df)
    rows = []

    for _, r in df.iterrows():
        group = group_seed_exp(str(r["Experiment_Folder"]))
        if group is not None:
            rows.append({
                "group": group,
                "experiment": r["Experiment_Folder"],
                "MCC": r.get("MCC", np.nan),
                "Balanced_Accuracy": r.get("Balanced_Accuracy", np.nan),
            })

    if not rows:
        placeholder(out_path, title, "seed experiments in all_experiments_summary.csv")
        return

    plot_df = pd.DataFrame(rows)
    groups = list(plot_df["group"].unique())
    x = np.arange(len(groups))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric in zip(axes, ["MCC", "Balanced_Accuracy"]):
        for i, g in enumerate(groups):
            vals = plot_df[plot_df["group"] == g][metric].dropna().values
            jitter = np.linspace(-0.06, 0.06, len(vals)) if len(vals) > 1 else [0]
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=80)
            if len(vals) > 0:
                ax.plot([i - 0.12, i + 0.12], [np.mean(vals), np.mean(vals)], linewidth=2)

        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=25, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    save_fig(out_path)


# =====================================================================
# Main
# =====================================================================

def make_archive(fig_dir: Path, archive_name: str):
    with tarfile.open(archive_name, "w:gz") as tar:
        tar.add(fig_dir, arcname=fig_dir.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig-dir", type=str, default="diploma_figures")
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--science-dir", type=str, default="science_folder")
    parser.add_argument("--model-path", type=str, default="science_folder/20_raw_supcon/best_model.pth")
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--archive-name", type=str, default="diploma_figures_fixed.tar.gz")
    args = parser.parse_args()

    fig_dir = Path(args.fig_dir)
    ch2 = fig_dir / "chapter_2"
    ch3 = fig_dir / "chapter_3"
    ch4 = fig_dir / "chapter_4"

    ensure_dir(ch2)
    ensure_dir(ch3)
    ensure_dir(ch4)

    dataset_root = Path(args.dataset_root)
    science_dir = Path(args.science_dir)
    model_path = Path(args.model_path)
    eval_dir = dataset_root / args.eval_split

    exp_df = read_csv_auto(Path("all_experiments_summary.csv"))
    exp_df = normalize_exp_df(exp_df)

    gan_best = read_csv_auto(Path("gan_analysis/gan_best_snapshots.csv"))

    print("[FIX] 2.4 / 2.5 real preprocessing examples")
    build_preprocessing_figures(dataset_root, ch2)

    print("[FIX] 3.4 dataset tree")
    draw_dataset_tree(ch3 / "fig_3_4_dataset_file_structure.png")

    print("[FIX] 3.1 / 3.2 / 3.5 / 3.8 multilevel architecture diagrams")
    multilevel_pipeline(
        ch3 / "fig_3_1_software_complex_architecture.png",
        "Общая архитектура исследовательского программного комплекса",
        {
            "Data layer": ["HAM10000", "lesion_id split", "raw / preprocessed"],
            "GAN layer": ["train only", "StyleGAN", "FID analysis", "synthetic images"],
            "Classifier layer": ["GAN-mix builder", "ConvNeXt-Large", "loss / optimizer", "best model"],
            "Evaluation layer": ["metrics", "calibration", "thresholds", "Grad-CAM", "aggregation"],
        },
    )

    multilevel_pipeline(
        ch3 / "fig_3_2_module_interaction.png",
        "Взаимодействие модулей в исследовательском пайплайне",
        {
            "Input": ["images", "metadata", "lesion_id"],
            "Dataset": ["split", "raw data", "preprocessed data"],
            "Generative branch": ["minority train", "GAN training", "best snapshot", "GAN-mix"],
            "Discriminative branch": ["dataloader", "ConvNeXt", "validation", "reports"],
            "Analysis": ["summary CSV", "Pareto", "calibration", "Grad-CAM"],
        },
    )

    multilevel_pipeline(
        ch3 / "fig_3_5_raw_preprocessed_variants.png",
        "Пайплайн подготовки raw и preprocessed вариантов датасета",
        {
            "Common split": ["dataset", "train", "valid", "test"],
            "Raw branch": ["no preprocessing", "same classes", "classifier train"],
            "Preprocessed branch": ["hair removal", "contrast normalization", "same split"],
            "GAN branch": ["train only", "crop/resize", "GAN-ready data"],
        },
    )

    multilevel_pipeline(
        ch3 / "fig_3_8_gan_mix_builder.png",
        "Формирование GAN-mix датасетов",
        {
            "Real data": ["real train", "real valid", "real test"],
            "Synthetic data": ["bcc", "akiec", "vasc", "df"],
            "Single-class mixes": ["bcc 25/50/100", "akiec 25/50", "vasc 25", "df 25"],
            "Combined mixes": ["bcc+akiec", "bcc+akiec+vasc", "valid/test unchanged"],
        },
    )

    print("[FIX] 3.13 / 4.23 standalone normalized confusion matrix")
    plot_confusion_matrix_from_model(
        model_path,
        eval_dir,
        [
            ch3 / "fig_3_13_normalized_confusion_matrix.png",
            ch4 / "fig_4_23_selected_model_confusion_matrix.png",
        ],
    )

    print("[FIX] 3.16 reliability diagram")
    plot_reliability_from_model(
        model_path,
        eval_dir,
        science_dir / "20_raw_supcon" / "calibration_report.json",
        ch3 / "fig_3_16_reliability_before_after.png",
    )

    print("[FIX] 4.16 grouped ECE before/after")
    plot_ece_grouped(science_dir, ch4 / "fig_4_16_ece_before_after_calibration.png")

    print("[FIX] threshold curves malignant/melanoma")
    plot_threshold_curves(
        science_dir,
        "malignant",
        [
            ch3 / "fig_3_17_threshold_malignant.png",
            ch4 / "fig_4_17_threshold_curves_malignant.png",
        ],
    )
    plot_threshold_curves(
        science_dir,
        "melanoma",
        [
            ch3 / "fig_3_18_threshold_melanoma.png",
            ch4 / "fig_4_18_threshold_curves_melanoma.png",
        ],
    )

    print("[FIX] 4.3 only 01_base_raw baseline")
    if exp_df is not None:
        baseline = exp_df[exp_df["Experiment_Folder"].isin(["01_base_raw"])].copy()
        if not baseline.empty:
            metrics = ["Accuracy", "Balanced_Accuracy", "MCC", "F1_Macro", "PR_AUC", "ROC_AUC_Macro"]
            vals = [float(baseline.iloc[0].get(m, np.nan)) for m in metrics]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(metrics, vals)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Metric value")
            ax.set_title("Интегральные метрики baseline real-only модели: 01_base_raw")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(True, axis="y", alpha=0.3)
            save_fig(ch4 / "fig_4_3_baseline_integral_metrics.png")
        else:
            placeholder(ch4 / "fig_4_3_baseline_integral_metrics.png", "Baseline 01_base_raw", "01_base_raw in all_experiments_summary.csv")

    print("[FIX] 4.7 filtered classical imbalance")
    scatter_filtered(
        exp_df,
        [
            "01_base_raw",
            "17_raw_weighted_ce",
            "18_raw_focal_g1.0",
            "19_raw_weighted_focal_g0.5",
            "21_raw_undersample_ce",
            "22_raw_oversample_ce",
            "23_raw_smooth_ce",
            "25_prep_focal_g1",
            "29_raw_focal_g0.5",
            "30_raw_focal_g2.0",
            "31_raw_focal_g2.0_weighted",
        ],
        "MCC",
        "mel_Recall",
        ch4 / "fig_4_7_classical_imbalance_methods.png",
        "Сравнение классических методов компенсации дисбаланса",
    )

    print("[FIX] 4.9 filtered SupCon")
    scatter_filtered(
        exp_df,
        [
            "01_base_raw",
            "20_raw_supcon",
            "20_raw_supcon_seed42",
            "20_raw_supcon_seed52",
            "20_raw_supcon_seed62",
            "24_raw_supcon_weighted",
            "32_prep_supcon_weighted",
            "33_gan_supcon_weighted",
            "ablation_raw_supcon",
        ],
        "MCC",
        "Balanced_Accuracy",
        ch4 / "fig_4_9_supcon_mcc_bacc_space.png",
        "SupCon-конфигурации в пространстве MCC и Balanced Accuracy",
    )

    print("[FIX] 4.13 filtered old GAN configs")
    scatter_filtered(
        exp_df,
        [
            "01_base_raw",
            "10_gan_baseline",
            "11_gan_smooth",
            "13_gan_supcon",
            "13_gan_supcon_seed42",
            "13_gan_supcon_seed52",
            "26_gan_weighted_ce",
            "27_gan_focal_g1",
            "28_gan_focal_g1_weighted",
            "33_gan_supcon_weighted",
        ],
        "MCC",
        "mel_Recall",
        ch4 / "fig_4_13_old_gan_configs_mcc_mel_recall.png",
        "Сравнение старых GAN-конфигураций по MCC и recall класса mel",
    )

    print("[FIX] 4.14 readable GAN-mix horizontal chart")
    plot_ganmix_horizontal(exp_df, ch4 / "fig_4_14_ganmix_metrics.png")

    print("[FIX] 4.15 FID vs BAcc scatter")
    plot_fid_vs_bacc(exp_df, gan_best, ch4 / "fig_4_15_fid_vs_ganmix_bacc.png")

    print("[FIX] 4.22 seed variability")
    plot_seed_variability(exp_df, ch4 / "fig_4_22_seed_variability.png")

    if args.archive:
        make_archive(fig_dir, args.archive_name)
        print(f"[SUCCESS] Archive created: {args.archive_name}")

    print("[SUCCESS] Figure fixes completed.")


if __name__ == "__main__":
    main()