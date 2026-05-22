#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Сборка датасетов для классификатора из:
1) базового датасета (dataset_preprocessed / dataset_preprocessed_v2),
2) синтетики, уже сгенерированной GAN,
3) плана экспериментов из gan_single_class_plan.csv / gan_combined_plan.csv.

Структура входного базового датасета:
    SRC_DATASET/
        train/<class>/*
        valid/<class>/*
        test/<class>/*    (опционально, но желательно)

Структура синтетики:
    SYN_ROOT/
        gan_bcc_25/bcc/*
        gan_bcc_50/bcc/*
        gan_bcc50_akiec50/bcc/*
        gan_bcc50_akiec50/akiec/*
        ...

Выход:
    OUT_ROOT/
        baseline_real_only/
        gan_bcc_25/
        gan_bcc_50/
        gan_bcc100_akiec50/
        ...
каждый с полной структурой:
    train / valid / test

Ключевой принцип:
- valid/test НЕ меняются;
- синтетика добавляется ТОЛЬКО в train/<class>.

Пример:
python build_gan_augmented_datasets.py \
    --src-dataset dataset_preprocessed_v2 \
    --synthetic-root synthetic_mixing_runs \
    --single-plan gan_analysis/gan_single_class_plan.csv \
    --combined-plan gan_analysis/gan_combined_plan.csv \
    --out-root datasets_gan_mix \
    --link-mode hardlink \
    --overwrite
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DATASET_CACHE_NAMES = {"ham_cache_u8.pt", "cache_uint8_256.pt"}


# ---------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------

def read_csv_semicolon(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)


def write_csv_semicolon(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def list_images(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.iterdir() if is_image_file(p)])


def safe_remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            safe_remove(path)
        else:
            raise FileExistsError(f"Папка уже существует: {path}")
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "symlink":
        os.symlink(src.resolve(), dst)
        return

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            # fallback
            shutil.copy2(src, dst)
            return

    raise ValueError(f"Неизвестный режим link-mode: {mode}")


def clone_dataset_tree(src_root: Path, dst_root: Path, mode: str) -> None:
    """
    Клонирует весь базовый датасет:
    train / valid / test и все файлы.
    """
    for root, _, files in os.walk(src_root):
        root_path = Path(root)
        rel = root_path.relative_to(src_root)
        dst_dir = dst_root / rel
        dst_dir.mkdir(parents=True, exist_ok=True)

        for fname in files:
            if fname in DATASET_CACHE_NAMES:
                continue
            src = root_path / fname
            dst = dst_dir / fname
            link_or_copy_file(src, dst, mode)


def remove_dataset_caches(dataset_root: Path) -> None:
    for cache_name in DATASET_CACHE_NAMES:
        for path in dataset_root.rglob(cache_name):
            path.unlink(missing_ok=True)


def parse_components(components: str) -> Dict[str, float]:
    """
    "bcc:0.50,akiec:0.50" -> {"bcc": 0.5, "akiec": 0.5}
    """
    out: Dict[str, float] = {}
    for chunk in components.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        cls, ratio = chunk.split(":")
        out[cls.strip()] = float(ratio.strip())
    return out


def count_train_classes(dataset_root: Path) -> Dict[str, int]:
    train_root = dataset_root / "train"
    out = {}
    if not train_root.exists():
        return out

    for cls_dir in sorted([p for p in train_root.iterdir() if p.is_dir()]):
        out[cls_dir.name] = len(list_images(cls_dir))
    return out


def load_single_plan(path: Path) -> List[dict]:
    rows = read_csv_semicolon(path)
    return rows


def load_combined_plan(path: Optional[Path]) -> List[dict]:
    if path is None or not path.exists():
        return []
    return read_csv_semicolon(path)


def build_experiment_spec(
    single_plan_rows: List[dict],
    combined_plan_rows: List[dict],
) -> Dict[str, Dict[str, float]]:
    """
    Возвращает карту:
    exp_name -> {class_name: ratio}
    baseline_real_only -> {}
    """
    exps: Dict[str, Dict[str, float]] = {}

    # single-class
    for row in single_plan_rows:
        exp_name = row["exp_name"].strip()
        if exp_name == "baseline_real_only":
            exps[exp_name] = {}
            continue

        cls = row["class_name"].strip()
        ratio = float(row["ratio"])
        exps[exp_name] = {cls: ratio}

    # combined
    for row in combined_plan_rows:
        exp_name = row["exp_name"].strip()
        exps[exp_name] = parse_components(row["components"])

    return exps


def validate_src_dataset(src_dataset: Path) -> None:
    if not src_dataset.exists():
        raise FileNotFoundError(f"Не найдена папка исходного датасета: {src_dataset}")
    if not (src_dataset / "train").exists():
        raise FileNotFoundError(f"Не найдена папка train в {src_dataset}")
    if not (src_dataset / "valid").exists():
        print(f"[WARN] В {src_dataset} нет valid/. train.py ожидает valid для валидации.")
    if not (src_dataset / "test").exists():
        print(f"[WARN] В {src_dataset} нет test/. Для честной финальной оценки test лучше сохранять.")


def copy_synthetic_into_train(
    exp_name: str,
    class_name: str,
    synth_dir: Path,
    dst_train_class_dir: Path,
    expected_count: Optional[int],
    strict: bool,
    mode: str,
) -> Tuple[int, List[dict]]:
    """
    Копирует/линкует синтетику в train/<class>.
    Добавляет префикс gan__ чтобы не было конфликтов с real filenames.
    """
    rows: List[dict] = []

    synth_imgs = list_images(synth_dir)
    if expected_count is not None and strict and len(synth_imgs) != expected_count:
        raise RuntimeError(
            f"[{exp_name}] {class_name}: ожидалось {expected_count} synthetic файлов, "
            f"но найдено {len(synth_imgs)} в {synth_dir}"
        )

    if expected_count is not None:
        synth_imgs = synth_imgs[:expected_count]

    dst_train_class_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for idx, src_img in enumerate(synth_imgs):
        stem = src_img.stem
        suffix = src_img.suffix.lower()
        dst_name = f"gan__{exp_name}__{class_name}__{idx:05d}__{stem}{suffix}"
        dst_img = dst_train_class_dir / dst_name

        link_or_copy_file(src_img, dst_img, mode)
        copied += 1

        rows.append(
            {
                "exp_name": exp_name,
                "class_name": class_name,
                "kind": "synthetic",
                "src_path": str(src_img),
                "dst_path": str(dst_img),
            }
        )

    return copied, rows


# ---------------------------------------------------------------------
# Основная логика
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--src-dataset", type=str, required=True,
                        help="Базовый датасет, например dataset_preprocessed_v2")
    parser.add_argument("--synthetic-root", type=str, required=True,
                        help="Корень с уже сгенерированной синтетикой, например synthetic_mixing_runs")
    parser.add_argument("--single-plan", type=str, required=True,
                        help="CSV из analyze_gan_fid_and_plan.py: gan_single_class_plan.csv")
    parser.add_argument("--combined-plan", type=str, default="",
                        help="CSV из analyze_gan_fid_and_plan.py: gan_combined_plan.csv (опционально)")
    parser.add_argument("--out-root", type=str, required=True,
                        help="Куда собирать итоговые датасеты")
    parser.add_argument("--link-mode", type=str, default="hardlink",
                        choices=["hardlink", "copy", "symlink"],
                        help="Как клонировать базовый датасет и synthetic файлы")
    parser.add_argument("--overwrite", action="store_true",
                        help="Пересобирать датасеты, если папки уже существуют")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Не собирать baseline_real_only")
    parser.add_argument("--strict", action="store_true",
                        help="Требовать точного совпадения числа synthetic файлов с планом")
    parser.add_argument("--experiments", nargs="*", default=[],
                        help="Явный список exp_name для сборки. Если пусто — собираются все.")
    args = parser.parse_args()

    src_dataset = Path(args.src_dataset)
    synthetic_root = Path(args.synthetic_root)
    single_plan_path = Path(args.single_plan)
    combined_plan_path = Path(args.combined_plan) if args.combined_plan else None
    out_root = Path(args.out_root)

    validate_src_dataset(src_dataset)

    single_rows = load_single_plan(single_plan_path)
    combined_rows = load_combined_plan(combined_plan_path)
    exp_specs = build_experiment_spec(single_rows, combined_rows)

    # Чтобы не потерять информацию о точном expected synthetic count
    single_lookup = {row["exp_name"]: row for row in single_rows}

    if args.experiments:
        selected = {name: spec for name, spec in exp_specs.items() if name in args.experiments}
    else:
        selected = exp_specs

    if args.skip_baseline and "baseline_real_only" in selected:
        selected.pop("baseline_real_only")

    out_root.mkdir(parents=True, exist_ok=True)

    global_summary_rows: List[dict] = []

    for exp_name, spec in selected.items():
        print("=" * 80)
        print(f"[BUILD] {exp_name}")
        print("=" * 80)

        dst_dataset = out_root / exp_name
        ensure_empty_dir(dst_dataset, overwrite=args.overwrite)

        # 1) Клонируем базовый датасет полностью
        print(f"[*] Клонирование базового датасета: {src_dataset} -> {dst_dataset}")
        clone_dataset_tree(src_dataset, dst_dataset, mode=args.link_mode)
        remove_dataset_caches(dst_dataset)

        manifest_rows: List[dict] = []

        # 2) Если baseline — на этом всё
        if exp_name == "baseline_real_only" or not spec:
            before_counts = count_train_classes(src_dataset)
            after_counts = count_train_classes(dst_dataset)

            for cls, cnt in sorted(after_counts.items()):
                global_summary_rows.append(
                    {
                        "exp_name": exp_name,
                        "class_name": cls,
                        "real_count": before_counts.get(cls, 0),
                        "synthetic_added": 0,
                        "final_count": cnt,
                    }
                )

            write_csv_semicolon(
                dst_dataset / "_gan_manifest.csv",
                manifest_rows,
                ["exp_name", "class_name", "kind", "src_path", "dst_path"],
            )
            continue

        # 3) Подмешиваем synthetic по классам
        before_counts = count_train_classes(src_dataset)
        synthetic_added_per_class: Dict[str, int] = {cls: 0 for cls in before_counts}

        for cls, ratio in spec.items():
            dst_train_class_dir = dst_dataset / "train" / cls
            synth_dir = synthetic_root / exp_name / cls

            if not synth_dir.exists():
                raise FileNotFoundError(
                    f"[{exp_name}] Не найдена папка synthetic для класса {cls}: {synth_dir}"
                )

            # expected_count берём из single-plan только для single-class экспериментов
            expected_count: Optional[int] = None
            if exp_name in single_lookup and single_lookup[exp_name]["class_name"] == cls:
                expected_count = int(single_lookup[exp_name]["synthetic_to_add"])

            copied, rows = copy_synthetic_into_train(
                exp_name=exp_name,
                class_name=cls,
                synth_dir=synth_dir,
                dst_train_class_dir=dst_train_class_dir,
                expected_count=expected_count,
                strict=args.strict,
                mode=args.link_mode,
            )

            synthetic_added_per_class[cls] = copied
            manifest_rows.extend(rows)

            print(f"[+] {cls}: добавлено synthetic = {copied}")

        # 4) Сводка
        after_counts = count_train_classes(dst_dataset)

        for cls in sorted(after_counts.keys()):
            global_summary_rows.append(
                {
                    "exp_name": exp_name,
                    "class_name": cls,
                    "real_count": before_counts.get(cls, 0),
                    "synthetic_added": synthetic_added_per_class.get(cls, 0),
                    "final_count": after_counts.get(cls, 0),
                }
            )

        write_csv_semicolon(
            dst_dataset / "_gan_manifest.csv",
            manifest_rows,
            ["exp_name", "class_name", "kind", "src_path", "dst_path"],
        )

        write_csv_semicolon(
            dst_dataset / "_gan_summary.csv",
            [
                {
                    "exp_name": exp_name,
                    "class_name": cls,
                    "real_count": before_counts.get(cls, 0),
                    "synthetic_added": synthetic_added_per_class.get(cls, 0),
                    "final_count": after_counts.get(cls, 0),
                }
                for cls in sorted(after_counts.keys())
            ],
            ["exp_name", "class_name", "real_count", "synthetic_added", "final_count"],
        )

    # 5) Общая сводка
    write_csv_semicolon(
        out_root / "_global_gan_dataset_summary.csv",
        global_summary_rows,
        ["exp_name", "class_name", "real_count", "synthetic_added", "final_count"],
    )

    print("\n[SUCCESS] Сборка датасетов завершена.")
    print(f"[*] Результат: {out_root.resolve()}")
    print(f"[*] Общая сводка: {out_root / '_global_gan_dataset_summary.csv'}")


if __name__ == "__main__":
    main()
