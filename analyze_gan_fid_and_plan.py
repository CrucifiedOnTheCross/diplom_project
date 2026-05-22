#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Анализ GAN-run'ов по metric-fid50k_full.jsonl и построение плана
подмешивания синтетики в классификатор.

Что делает:
1. Рекурсивно ищет metric-fid50k_full.jsonl в указанной папке.
2. Парсит FID-кривые.
3. Для каждого run выбирает лучший checkpoint по минимуму FID.
4. Определяет, вышел ли run на плато.
5. Строит:
   - gan_fid_history.csv
   - gan_fid_summary.csv
   - gan_best_snapshots.csv
   - gan_single_class_plan.csv
   - gan_combined_plan.csv
   - generate_commands.sh

Пример:
python analyze_gan_fid_and_plan.py \
    --root gan_training_runs_transfer \
    --out-dir gan_analysis \
    --trunc 0.7

Если train-counts отличаются от значений по умолчанию, их можно передать:
python analyze_gan_fid_and_plan.py \
    --root gan_training_runs_transfer \
    --out-dir gan_analysis \
    --train-count bcc=366 \
    --train-count akiec=230 \
    --train-count vasc=99 \
    --train-count df=115
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------
# Настройки по умолчанию
# ---------------------------------------------------------------------

# Это НЕ размеры zip для GAN после xflip, а реальные размеры train-классов
# классификатора, к которым будет подмешиваться синтетика.
DEFAULT_TRAIN_COUNTS: Dict[str, int] = {
    "bcc": 366,
    "akiec": 230,
    "vasc": 99,
    "df": 115,
}

# Пороговые зоны качества по best FID
# Эти зоны нужны не как "абсолютная истина", а как практическая эвристика.
FID_POLICY = [
    ("strong", 50.0, [0.25, 0.50, 1.00]),
    ("moderate", 65.0, [0.25, 0.50]),
    ("cautious", 80.0, [0.25]),
    ("ablation_only", 90.0, [0.25]),
    ("do_not_mix", float("inf"), []),
]


@dataclass
class FIDPoint:
    run_name: str
    class_name: str
    metric_file: str
    snapshot_pkl: str
    snapshot_kimg: int
    fid: float
    total_time_sec: Optional[float]


@dataclass
class RunSummary:
    run_name: str
    class_name: str
    metric_file: str
    num_points: int
    first_snapshot: str
    first_kimg: int
    first_fid: float
    best_snapshot: str
    best_kimg: int
    best_fid: float
    final_snapshot: str
    final_kimg: int
    final_fid: float
    relative_improvement_pct: float
    absolute_improvement: float
    recent_gain: float
    plateau: bool
    quality_band: str
    recommended_ratios: str


def parse_train_counts(items: List[str]) -> Dict[str, int]:
    counts = dict(DEFAULT_TRAIN_COUNTS)
    for item in items:
        if "=" not in item:
            raise ValueError(f"Некорректный --train-count: {item}. Ожидается class=count")
        cls, val = item.split("=", 1)
        cls = cls.strip()
        val = int(val.strip())
        counts[cls] = val
    return counts


def infer_class_from_path(path: Path) -> str:
    s = str(path)

    patterns = [
        r"stylegan2-([a-zA-Z0-9_]+)_raw_256",
        r"raw_([a-zA-Z0-9_]+)_stylegan2",
        r"/([a-zA-Z0-9_]+)_raw_256",
    ]

    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return m.group(1).lower()

    return "unknown"


def infer_run_name(metric_path: Path) -> str:
    # Родитель metric-файла обычно вида:
    # 00000-stylegan2-bcc_raw_256-gpus1-batch16-gamma0.8192
    return metric_path.parent.name


def snapshot_to_kimg(snapshot_name: str) -> int:
    # network-snapshot-000600.pkl -> 600
    m = re.search(r"network-snapshot-(\d+)\.pkl", snapshot_name)
    if not m:
        raise ValueError(f"Не удалось извлечь kimg из {snapshot_name}")
    return int(m.group(1))


def choose_policy(best_fid: float) -> Tuple[str, List[float]]:
    for band, thr, ratios in FID_POLICY:
        if best_fid <= thr:
            return band, ratios
    return "do_not_mix", []


def detect_plateau(points: List[FIDPoint], recent_window: int = 4, min_gain: float = 1.0) -> Tuple[bool, float]:
    """
    Простая эвристика:
    сравниваем лучший FID на предыдущем окне и лучший FID на последнем окне.
    Если gain < min_gain, считаем, что run близок к плато.
    """
    if len(points) < recent_window * 2:
        return False, float("nan")

    prev = points[-2 * recent_window : -recent_window]
    recent = points[-recent_window:]

    prev_best = min(p.fid for p in prev)
    recent_best = min(p.fid for p in recent)
    gain = prev_best - recent_best

    return gain < min_gain, gain


def parse_metric_jsonl(metric_path: Path) -> List[FIDPoint]:
    class_name = infer_class_from_path(metric_path)
    run_name = infer_run_name(metric_path)
    points: List[FIDPoint] = []

    with metric_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            results = obj.get("results", {})
            if "fid50k_full" not in results:
                continue

            snapshot = Path(obj["snapshot_pkl"]).name
            kimg = snapshot_to_kimg(snapshot)
            fid = float(results["fid50k_full"])
            total_time = obj.get("total_time", None)
            total_time = float(total_time) if total_time is not None else None

            points.append(
                FIDPoint(
                    run_name=run_name,
                    class_name=class_name,
                    metric_file=str(metric_path),
                    snapshot_pkl=snapshot,
                    snapshot_kimg=kimg,
                    fid=fid,
                    total_time_sec=total_time,
                )
            )

    points.sort(key=lambda x: x.snapshot_kimg)
    return points


def summarize_run(points: List[FIDPoint]) -> RunSummary:
    if not points:
        raise ValueError("Пустой список points")

    first = points[0]
    best = min(points, key=lambda x: x.fid)
    final = points[-1]

    rel_impr = 100.0 * (first.fid - best.fid) / first.fid if first.fid > 0 else 0.0
    abs_impr = first.fid - best.fid

    plateau, recent_gain = detect_plateau(points)
    band, ratios = choose_policy(best.fid)

    return RunSummary(
        run_name=first.run_name,
        class_name=first.class_name,
        metric_file=first.metric_file,
        num_points=len(points),
        first_snapshot=first.snapshot_pkl,
        first_kimg=first.snapshot_kimg,
        first_fid=first.fid,
        best_snapshot=best.snapshot_pkl,
        best_kimg=best.snapshot_kimg,
        best_fid=best.fid,
        final_snapshot=final.snapshot_pkl,
        final_kimg=final.snapshot_kimg,
        final_fid=final.fid,
        relative_improvement_pct=rel_impr,
        absolute_improvement=abs_impr,
        recent_gain=recent_gain,
        plateau=plateau,
        quality_band=band,
        recommended_ratios=",".join(f"{r:.2f}" for r in ratios) if ratios else "",
    )


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_single_class_plan(best_by_class: Dict[str, RunSummary], train_counts: Dict[str, int]) -> List[dict]:
    rows: List[dict] = []

    rows.append(
        {
            "exp_name": "baseline_real_only",
            "class_name": "ALL",
            "ratio": 0.0,
            "real_count": "",
            "synthetic_to_add": 0,
            "final_count": "",
            "snapshot_pkl": "",
            "best_fid": "",
            "priority": 0,
            "comment": "Базовый эксперимент без GAN-синтетики",
        }
    )

    priority = 1

    # сортировка классов по качеству генератора
    ordered = sorted(best_by_class.values(), key=lambda x: x.best_fid)

    for summary in ordered:
        cls = summary.class_name
        if cls not in train_counts:
            continue

        _, ratios = choose_policy(summary.best_fid)
        real_n = train_counts[cls]

        for ratio in ratios:
            synth_n = int(math.ceil(real_n * ratio))
            rows.append(
                {
                    "exp_name": f"gan_{cls}_{int(ratio * 100)}",
                    "class_name": cls,
                    "ratio": ratio,
                    "real_count": real_n,
                    "synthetic_to_add": synth_n,
                    "final_count": real_n + synth_n,
                    "snapshot_pkl": summary.best_snapshot,
                    "best_fid": f"{summary.best_fid:.6f}",
                    "priority": priority,
                    "comment": f"{summary.quality_band}; best @ {summary.best_kimg} kimg",
                }
            )
            priority += 1

    return rows


def build_combined_plan(best_by_class: Dict[str, RunSummary], train_counts: Dict[str, int]) -> List[dict]:
    rows: List[dict] = []

    # Умеренные комбинации: сначала лучшие генераторы
    def exists(cls: str) -> bool:
        return cls in best_by_class and cls in train_counts

    # bcc + akiec
    if exists("bcc") and exists("akiec"):
        rows.append(
            {
                "exp_name": "gan_bcc50_akiec50",
                "components": "bcc:0.50,akiec:0.50",
                "comment": "Основная умеренная комбинация двух лучших генераторов",
            }
        )
        rows.append(
            {
                "exp_name": "gan_bcc100_akiec50",
                "components": "bcc:1.00,akiec:0.50",
                "comment": "Смещение в сторону лучшего генератора bcc",
            }
        )

    # Добавляем vasc только если он не совсем плохой
    if exists("bcc") and exists("akiec") and exists("vasc"):
        if best_by_class["vasc"].best_fid <= 80.0:
            rows.append(
                {
                    "exp_name": "gan_bcc50_akiec50_vasc25",
                    "components": "bcc:0.50,akiec:0.50,vasc:0.25",
                    "comment": "Аккуратная тройная смесь с мягким добавлением vasc",
                }
            )

    # df включать в multi-class mix только если FID стал приемлемым
    if exists("bcc") and exists("df"):
        if best_by_class["df"].best_fid <= 80.0:
            rows.append(
                {
                    "exp_name": "gan_bcc50_df25",
                    "components": "bcc:0.50,df:0.25",
                    "comment": "Контрольная смесь при улучшении качества df",
                }
            )

    return rows


def build_generation_commands(
    out_path: Path,
    single_plan: List[dict],
    combined_plan: List[dict],
    best_by_class: Dict[str, RunSummary],
    train_counts: Dict[str, int],
    trunc: float,
    gen_script: str,
    synthetic_root: str,
) -> None:
    """
    Генерирует bash-скрипт с командами gen_images.py
    под single-class и combined эксперименты.
    """
    lines: List[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'GEN_SCRIPT="{gen_script}"',
        f'SYN_ROOT="{synthetic_root}"',
        "",
        'mkdir -p "$SYN_ROOT"',
        "",
    ]

    lines.append("# =========================================================")
    lines.append("# SINGLE-CLASS EXPERIMENTS")
    lines.append("# =========================================================")

    for row in single_plan:
        if row["exp_name"] == "baseline_real_only":
            continue

        cls = row["class_name"]
        synth_n = int(row["synthetic_to_add"])
        if synth_n <= 0:
            continue

        summary = best_by_class[cls]
        outdir = f'$SYN_ROOT/{row["exp_name"]}/{cls}'
        seeds = f"0-{synth_n - 1}"

        lines += [
            f'echo "[RUN] {row["exp_name"]}"',
            f'mkdir -p "{outdir}"',
            "python \"$GEN_SCRIPT\" \\",
            f'  --network="{Path(summary.metric_file).parent / summary.best_snapshot}" \\',
            f'  --outdir="{outdir}" \\',
            f'  --seeds="{seeds}" \\',
            f'  --trunc={trunc}',
            "",
        ]

    lines.append("# =========================================================")
    lines.append("# COMBINED EXPERIMENTS")
    lines.append("# =========================================================")

    for row in combined_plan:
        exp_name = row["exp_name"]
        components = row["components"].split(",")

        lines.append(f'echo "[RUN] {exp_name}"')
        for comp in components:
            cls, ratio_str = comp.split(":")
            cls = cls.strip()
            ratio = float(ratio_str.strip())
            if cls not in train_counts or cls not in best_by_class:
                continue

            synth_n = int(math.ceil(train_counts[cls] * ratio))
            if synth_n <= 0:
                continue

            summary = best_by_class[cls]
            outdir = f'$SYN_ROOT/{exp_name}/{cls}'
            seeds = f"0-{synth_n - 1}"

            lines += [
                f'mkdir -p "{outdir}"',
                "python \"$GEN_SCRIPT\" \\",
                f'  --network="{Path(summary.metric_file).parent / summary.best_snapshot}" \\',
                f'  --outdir="{outdir}" \\',
                f'  --seeds="{seeds}" \\',
                f'  --trunc={trunc}',
                "",
            ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_markdown_report(
    out_path: Path,
    summaries: List[RunSummary],
    best_by_class: Dict[str, RunSummary],
) -> None:
    lines: List[str] = []
    lines.append("# GAN FID analysis report")
    lines.append("")
    lines.append("## Ranking by best FID")
    lines.append("")

    ranked = sorted(best_by_class.values(), key=lambda x: x.best_fid)
    for i, s in enumerate(ranked, start=1):
        lines.append(
            f"{i}. **{s.class_name}** — best FID={s.best_fid:.4f} "
            f"at {s.best_kimg} kimg ({s.best_snapshot}), "
            f"band={s.quality_band}, plateau={s.plateau}"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `strong`: можно тестировать +25%, +50%, +100%")
    lines.append("- `moderate`: +25%, +50%")
    lines.append("- `cautious`: только +25%")
    lines.append("- `ablation_only`: только +25% как контроль")
    lines.append("- `do_not_mix`: не подмешивать в основную серию")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Корень с GAN run'ами")
    parser.add_argument("--out-dir", type=str, default="gan_analysis", help="Папка для выходных файлов")
    parser.add_argument("--trunc", type=float, default=0.7, help="trunc для gen_images.py")
    parser.add_argument(
        "--gen-script",
        type=str,
        default="stylegan3-brecahad/stylegan3/gen_images.py",
        help="Путь к gen_images.py",
    )
    parser.add_argument(
        "--synthetic-root",
        type=str,
        default="synthetic_mixing_runs",
        help="Куда генерировать синтетику",
    )
    parser.add_argument(
        "--train-count",
        action="append",
        default=[],
        help="Размер real train-класса, формат class=count. Можно повторять.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_counts = parse_train_counts(args.train_count)

    metric_files = sorted(root.rglob("metric-fid50k_full.jsonl"))
    if not metric_files:
        raise FileNotFoundError(f"Не найдено ни одного metric-fid50k_full.jsonl в {root}")

    all_points: List[FIDPoint] = []
    summaries: List[RunSummary] = []

    for metric_file in metric_files:
        points = parse_metric_jsonl(metric_file)
        if not points:
            continue
        all_points.extend(points)
        summaries.append(summarize_run(points))

    if not summaries:
        raise RuntimeError("Не удалось распарсить ни одного GAN-run")

    # лучшая сводка по каждому классу
    best_by_class: Dict[str, RunSummary] = {}
    for s in summaries:
        if s.class_name not in best_by_class or s.best_fid < best_by_class[s.class_name].best_fid:
            best_by_class[s.class_name] = s

    # 1) история
    history_rows = [
        {
            "run_name": p.run_name,
            "class_name": p.class_name,
            "metric_file": p.metric_file,
            "snapshot_pkl": p.snapshot_pkl,
            "snapshot_kimg": p.snapshot_kimg,
            "fid": f"{p.fid:.6f}",
            "total_time_sec": "" if p.total_time_sec is None else f"{p.total_time_sec:.3f}",
        }
        for p in all_points
    ]
    write_csv(
        out_dir / "gan_fid_history.csv",
        history_rows,
        ["run_name", "class_name", "metric_file", "snapshot_pkl", "snapshot_kimg", "fid", "total_time_sec"],
    )

    # 2) summary
    summary_rows = [asdict(s) for s in summaries]
    write_csv(
        out_dir / "gan_fid_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()),
    )

    # 3) best snapshots
    best_rows = [asdict(s) for s in sorted(best_by_class.values(), key=lambda x: x.best_fid)]
    write_csv(
        out_dir / "gan_best_snapshots.csv",
        best_rows,
        list(best_rows[0].keys()),
    )

    # 4) single-class plan
    single_plan = build_single_class_plan(best_by_class, train_counts)
    write_csv(
        out_dir / "gan_single_class_plan.csv",
        single_plan,
        ["exp_name", "class_name", "ratio", "real_count", "synthetic_to_add", "final_count",
         "snapshot_pkl", "best_fid", "priority", "comment"],
    )

    # 5) combined plan
    combined_plan = build_combined_plan(best_by_class, train_counts)
    if combined_plan:
        write_csv(
            out_dir / "gan_combined_plan.csv",
            combined_plan,
            ["exp_name", "components", "comment"],
        )

    # 6) generation commands
    build_generation_commands(
        out_path=out_dir / "generate_commands.sh",
        single_plan=single_plan,
        combined_plan=combined_plan,
        best_by_class=best_by_class,
        train_counts=train_counts,
        trunc=args.trunc,
        gen_script=args.gen_script,
        synthetic_root=args.synthetic_root,
    )

    # 7) markdown report
    build_markdown_report(
        out_path=out_dir / "gan_report.md",
        summaries=summaries,
        best_by_class=best_by_class,
    )

    print("=" * 70)
    print("[DONE] GAN analysis finished")
    print(f"[OUT ] {out_dir.resolve()}")
    print("=" * 70)

    print("\n[TOP GANs by best FID]")
    for s in sorted(best_by_class.values(), key=lambda x: x.best_fid):
        print(
            f"  {s.class_name:>6} | best FID={s.best_fid:.4f} "
            f"@ {s.best_kimg:>4} kimg | {s.best_snapshot} | "
            f"band={s.quality_band} | plateau={s.plateau}"
        )

    print("\n[Single-class plan preview]")
    for row in single_plan[:10]:
        print(" ", row)

    if combined_plan:
        print("\n[Combined plan preview]")
        for row in combined_plan:
            print(" ", row)


if __name__ == "__main__":
    main()