#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from model import JointSkinLesionClassifier

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


def load_model(model_path: str, device: str = "cuda"):
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    model = JointSkinLesionClassifier(num_classes=len(CLASS_NAMES)).to(device_obj)
    state = torch.load(model_path, map_location=device_obj)

    # fix for torch.compile checkpoints
    cleaned = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            cleaned[k.replace("_orig_mod.", "")] = v
        else:
            cleaned[k] = v

    model.load_state_dict(cleaned)
    model.eval()
    return model, device_obj


def find_target_layers(model: torch.nn.Module):
    return [model.backbone.features[-1][-1]]


def preprocess_image(img_path: str, device_obj: torch.device):
    img = Image.open(img_path).convert("RGB")
    rgb_img = np.float32(img.resize((256, 256))) / 255.0

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0).to(device_obj)
    return rgb_img, tensor


@torch.no_grad()
def predict_one(model: torch.nn.Module, img_tensor: torch.Tensor):
    logits = model(img_tensor, return_embeddings=False)
    probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def compute_gradcam(model, img_path: str, device_obj, target_idx: Optional[int] = None):
    rgb_img, input_tensor = preprocess_image(img_path, device_obj)
    pred_idx, probs = predict_one(model, input_tensor)

    if target_idx is None:
        target_idx = pred_idx

    cam = GradCAM(model=model, target_layers=find_target_layers(model))
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(target_idx)]
    )[0, :]
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return rgb_img, overlay, pred_idx, probs


def get_images_by_class(split_dir: str) -> Dict[str, List[Path]]:
    root = Path(split_dir)
    if not root.exists():
        raise FileNotFoundError(f"Папка не найдена: {root}")

    out = {}
    for cls in CLASS_NAMES:
        class_dir = root / cls
        images = sorted(
            list(class_dir.glob("*.jpg")) +
            list(class_dir.glob("*.jpeg")) +
            list(class_dir.glob("*.png"))
        )
        out[cls] = images
    return out


def collect_predictions(model, split_dir: str, device_obj):
    rows = []
    by_class = get_images_by_class(split_dir)

    for true_cls in CLASS_NAMES:
        for img_path in by_class[true_cls]:
            _, tensor = preprocess_image(str(img_path), device_obj)
            pred_idx, probs = predict_one(model, tensor)

            pred_cls = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])
            true_prob = float(probs[CLASS_NAMES.index(true_cls)])

            rows.append({
                "img_path": str(img_path),
                "file_name": img_path.name,
                "true_cls": true_cls,
                "pred_cls": pred_cls,
                "correct": int(pred_cls == true_cls),
                "pred_confidence": confidence,
                "true_class_probability": true_prob,
            })
    return rows


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def build_summary_by_class(rows: List[dict]):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["true_cls"]].append(r)

    summary = []
    for cls in CLASS_NAMES:
        items = grouped[cls]
        total = len(items)
        correct = sum(x["correct"] for x in items)
        wrong = total - correct
        acc = correct / total if total else 0.0

        summary.append({
            "class": cls,
            "n_total": total,
            "n_correct": correct,
            "n_wrong": wrong,
            "class_accuracy": f"{acc:.6f}",
            "has_correct_examples": int(correct > 0),
            "has_wrong_examples": int(wrong > 0),
        })
    return summary


def build_error_pairs(rows: List[dict]):
    counter = Counter()
    conf_sum = Counter()

    for r in rows:
        if not r["correct"]:
            key = (r["true_cls"], r["pred_cls"])
            counter[key] += 1
            conf_sum[key] += r["pred_confidence"]

    out = []
    for (true_cls, pred_cls), cnt in sorted(counter.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
        out.append({
            "true_cls": true_cls,
            "pred_cls": pred_cls,
            "count": cnt,
            "mean_pred_confidence": f"{(conf_sum[(true_cls, pred_cls)] / cnt):.6f}",
        })
    return out


def select_examples(rows: List[dict], per_kind: int = 2):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["true_cls"]].append(r)

    selected = []
    for cls in CLASS_NAMES:
        items = grouped[cls]

        correct = sorted(
            [x for x in items if x["correct"]],
            key=lambda x: (-x["pred_confidence"], x["img_path"])
        )[:per_kind]

        wrong = sorted(
            [x for x in items if not x["correct"]],
            key=lambda x: (-x["pred_confidence"], x["img_path"])
        )[:per_kind]

        for rank, ex in enumerate(correct, start=1):
            selected.append({
                **ex,
                "kind": "correct",
                "rank_within_kind": rank,
            })
        for rank, ex in enumerate(wrong, start=1):
            selected.append({
                **ex,
                "kind": "wrong",
                "rank_within_kind": rank,
            })

    selected = sorted(selected, key=lambda x: (x["true_cls"], x["kind"], x["rank_within_kind"]))
    return selected


def save_side_by_side(model, selected_rows: List[dict], out_dir: Path, device_obj):
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_rows = []

    for row in selected_rows:
        rgb_img, overlay, pred_idx, probs = compute_gradcam(model, row["img_path"], device_obj)
        pred_cls = CLASS_NAMES[pred_idx]
        conf = float(probs[pred_idx]) * 100.0

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Оригинал\ntrue={row['true_cls']}")
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM\npred={pred_cls} | {conf:.1f}%")
        axes[1].axis("off")

        plt.tight_layout()

        save_name = (
            f"{row['kind']}__{row['true_cls']}__rank{row['rank_within_kind']}__"
            f"pred_{row['pred_cls']}__{Path(row['img_path']).stem}.png"
        )
        save_path = out_dir / save_name
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        saved_rows.append({
            **row,
            "saved_visual_path": str(save_path),
        })

    return saved_rows


def _add_panel(ax, img_path: Optional[str], title: str):
    ax.axis("off")
    if not img_path or not Path(img_path).exists():
        ax.text(0.5, 0.5, "Не найдено", ha="center", va="center", fontsize=11)
        ax.set_title(title, fontsize=10)
        return

    img = Image.open(img_path).convert("RGB")
    ax.imshow(img)
    ax.set_title(title, fontsize=9)


def build_kind_grid(saved_rows: List[dict], out_path: Path, kind: str, per_kind: int):
    rows_n = len(CLASS_NAMES)
    cols_n = per_kind

    fig, axes = plt.subplots(rows_n, cols_n, figsize=(5 * cols_n, 3.8 * rows_n))
    if rows_n == 1:
        axes = np.array([axes])
    if cols_n == 1:
        axes = axes.reshape(rows_n, 1)

    index = defaultdict(dict)
    for row in saved_rows:
        if row["kind"] == kind:
            index[row["true_cls"]][row["rank_within_kind"]] = row

    for i, cls in enumerate(CLASS_NAMES):
        for j in range(1, per_kind + 1):
            row = index[cls].get(j)
            img_path = row["saved_visual_path"] if row else None
            title = f"{cls} | {kind} | rank={j}"
            _add_panel(axes[i, j - 1], img_path, title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_pair_grid(saved_rows: List[dict], out_path: Path):
    fig, axes = plt.subplots(len(CLASS_NAMES), 2, figsize=(12, 3.8 * len(CLASS_NAMES)))
    if len(CLASS_NAMES) == 1:
        axes = np.array([axes])

    index = defaultdict(dict)
    for row in saved_rows:
        if row["rank_within_kind"] == 1:
            index[row["true_cls"]][row["kind"]] = row

    for i, cls in enumerate(CLASS_NAMES):
        corr = index[cls].get("correct")
        wrong = index[cls].get("wrong")

        _add_panel(
            axes[i, 0],
            corr["saved_visual_path"] if corr else None,
            f"{cls}: правильный пример"
        )
        _add_panel(
            axes[i, 1],
            wrong["saved_visual_path"] if wrong else None,
            f"{cls}: ошибочный пример"
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_manual_review_template(saved_rows: List[dict]):
    out = []
    for row in saved_rows:
        out.append({
            "kind": row["kind"],
            "true_cls": row["true_cls"],
            "pred_cls": row["pred_cls"],
            "rank_within_kind": row["rank_within_kind"],
            "pred_confidence": f"{row['pred_confidence']:.6f}",
            "img_path": row["img_path"],
            "saved_visual_path": row["saved_visual_path"],
            "focus_on_lesion_core": "",
            "focus_on_lesion_border": "",
            "focus_on_background": "",
            "focus_on_hair_or_glare": "",
            "focus_on_peripheral_skin": "",
            "manual_comment": "",
        })
    return out


def main():
    model_path = "science_folder/20_raw_supcon/best_model.pth"
    split_dir = "dataset/test"
    out_root = Path("gradcam_diploma")
    selected_dir = out_root / "selected_examples"
    out_root.mkdir(exist_ok=True, parents=True)

    device = "cuda"
    per_kind = 2

    print(f"[*] Загрузка модели: {model_path}")
    model, device_obj = load_model(model_path, device=device)

    print(f"[*] Сбор предсказаний по выборке: {split_dir}")
    all_rows = collect_predictions(model, split_dir, device_obj)

    print("[*] Построение таблиц...")
    summary_rows = build_summary_by_class(all_rows)
    error_pair_rows = build_error_pairs(all_rows)
    selected_rows = select_examples(all_rows, per_kind=per_kind)

    print("[*] Сохранение выбранных визуализаций...")
    saved_rows = save_side_by_side(model, selected_rows, selected_dir, device_obj)

    print("[*] Построение сеток...")
    build_pair_grid(saved_rows, out_root / "gradcam_grid_pair.png")
    build_kind_grid(saved_rows, out_root / "gradcam_grid_correct.png", kind="correct", per_kind=per_kind)
    build_kind_grid(saved_rows, out_root / "gradcam_grid_wrong.png", kind="wrong", per_kind=per_kind)

    print("[*] Сохранение CSV...")
    write_csv(
        out_root / "gradcam_examples_all.csv",
        all_rows,
        ["img_path", "file_name", "true_cls", "pred_cls", "correct",
         "pred_confidence", "true_class_probability"]
    )

    write_csv(
        out_root / "gradcam_summary_by_class.csv",
        summary_rows,
        ["class", "n_total", "n_correct", "n_wrong",
         "class_accuracy", "has_correct_examples", "has_wrong_examples"]
    )

    write_csv(
        out_root / "gradcam_error_pairs.csv",
        error_pair_rows,
        ["true_cls", "pred_cls", "count", "mean_pred_confidence"]
    )

    write_csv(
        out_root / "gradcam_selected_examples.csv",
        saved_rows,
        ["img_path", "file_name", "true_cls", "pred_cls", "correct",
         "pred_confidence", "true_class_probability",
         "kind", "rank_within_kind", "saved_visual_path"]
    )

    manual_review_rows = build_manual_review_template(saved_rows)
    write_csv(
        out_root / "gradcam_manual_review_template.csv",
        manual_review_rows,
        ["kind", "true_cls", "pred_cls", "rank_within_kind",
         "pred_confidence", "img_path", "saved_visual_path",
         "focus_on_lesion_core", "focus_on_lesion_border",
         "focus_on_background", "focus_on_hair_or_glare",
         "focus_on_peripheral_skin", "manual_comment"]
    )

    print("\n[SUCCESS] Grad-CAM diploma package is ready.")
    print(f"[*] Output dir: {out_root.resolve()}")


if __name__ == "__main__":
    main()