#!/usr/bin/env python3
"""Run predefined model significance comparisons and aggregate reports."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd


PAIRS = [
    ("ablation_raw_focal_g1_weighted", "24_raw_supcon_weighted"),
    ("ablation_raw_focal_g1_weighted", "gan_bcc50_akiec50_vasc25_raw_ganmix_focal_weighted"),
    ("gan_bcc50_akiec50_vasc25_raw_ganmix", "gan_bcc50_akiec50_vasc25_raw_ganmix_focal_weighted"),
    ("random_ganmix_25_raw_ce", "random_ganmix_25_focal_weighted"),
    ("ablation_raw_focal_g1_weighted", "ensemble_real_supcon_ganmix"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run predefined statistical significance comparisons.")
    parser.add_argument("--science-dir", default="science_folder")
    parser.add_argument("--out-dir", default="science_folder/statistical_significance")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--fast", action="store_true", help="Use 2000 bootstrap and 2000 permutation iterations")
    parser.add_argument("--final", action="store_true", help="Use 10000 bootstrap and 10000 permutation iterations")
    parser.add_argument("--force", action="store_true", help="Recompute pairwise CSV files even when they are up to date")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Compute only missing or outdated pairwise CSV files. This is the default when --force is not used.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Number of pairwise comparisons to run in parallel")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.fast and args.final:
        parser.error("--fast and --final are mutually exclusive")
    if args.fast:
        args.n_bootstrap = 2000
        args.n_permutations = 2000
    if args.final:
        args.n_bootstrap = 10000
        args.n_permutations = 10000
        args.force = True
    return args


def prediction_file(science_dir: Path, experiment: str) -> Path | None:
    exp_dir = science_dir / experiment
    candidates = [exp_dir / "test_predictions.csv", exp_dir / "ensemble_predictions.csv"]
    for path in candidates:
        if path.exists():
            return path
    return None


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def output_is_current(out_csv: Path, dependencies: List[Path]) -> bool:
    if not out_csv.exists() or not dependencies or not all(path.exists() for path in dependencies):
        return False
    newest_dependency = max(path.stat().st_mtime for path in dependencies)
    return out_csv.stat().st_mtime >= newest_dependency


def compare_pair(task: dict) -> dict:
    name_a = task["name_a"]
    name_b = task["name_b"]
    pred_a = Path(task["pred_a"])
    pred_b = Path(task["pred_b"])
    out_csv = Path(task["out_csv"])

    if output_is_current(out_csv, [pred_a, pred_b]) and not task["force"]:
        return {"status": "ok", "out_csv": str(out_csv), "model_a": name_a, "model_b": name_b, "detail": "up to date"}

    cmd = [
        sys.executable,
        "compare_model_significance.py",
        "--a",
        str(pred_a),
        "--b",
        str(pred_b),
        "--name-a",
        name_a,
        "--name-b",
        name_b,
        "--out",
        str(out_csv),
        "--n-bootstrap",
        str(task["n_bootstrap"]),
        "--n-permutations",
        str(task["n_permutations"]),
        "--seed",
        str(task["seed"]),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {
            "status": "skipped",
            "model_a": name_a,
            "model_b": name_b,
            "reason": "comparison failed",
            "detail": (result.stderr or result.stdout).strip(),
        }
    return {"status": "ok", "out_csv": str(out_csv), "model_a": name_a, "model_b": name_b, "detail": "computed"}


def write_markdown(results: pd.DataFrame, skipped: pd.DataFrame, out_path: Path) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        headers = list(df.columns)
        rows = df.astype(str).values.tolist()
        out = []
        out.append("| " + " | ".join(headers) + " |")
        out.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            out.append("| " + " | ".join(str(cell) for cell in row) + " |")
        return "\n".join(out)

    lines: List[str] = []
    lines.append("# Pairwise Statistical Significance")
    lines.append("")
    lines.append("Различие считается статистически значимым, если `p < 0.05` и 95% CI для разности не содержит 0.")
    lines.append("")

    if results.empty:
        lines.append("Нет успешно сравненных пар.")
    else:
        for (model_a, model_b), group in results.groupby(["model_a", "model_b"], sort=False):
            lines.append(f"## {model_a} vs {model_b}")
            lines.append("")
            table = group[[
                "metric",
                "value_a",
                "value_b",
                "delta_b_minus_a",
                "ci95_low",
                "ci95_high",
                "permutation_p_value",
                "significant_p05",
                "ci_excludes_zero",
            ]].copy()
            for col in ["value_a", "value_b", "delta_b_minus_a", "ci95_low", "ci95_high", "permutation_p_value"]:
                table[col] = pd.to_numeric(table[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
            lines.append(markdown_table(table))
            lines.append("")

    if not skipped.empty:
        lines.append("## Skipped Pairs")
        lines.append("")
        lines.append(markdown_table(skipped))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    science_dir = Path(args.science_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_frames = []
    skipped_rows = []
    tasks = []

    for idx, (name_a, name_b) in enumerate(PAIRS, start=1):
        pred_a = prediction_file(science_dir, name_a)
        pred_b = prediction_file(science_dir, name_b)

        if pred_a is None or pred_b is None:
            skipped_rows.append({
                "model_a": name_a,
                "model_b": name_b,
                "reason": "missing prediction file",
                "detail": f"a={pred_a}, b={pred_b}",
            })
            continue

        out_csv = out_dir / f"significance_{idx:02d}_{safe_name(name_a)}_vs_{safe_name(name_b)}.csv"
        tasks.append({
            "name_a": name_a,
            "name_b": name_b,
            "pred_a": str(pred_a),
            "pred_b": str(pred_b),
            "out_csv": str(out_csv),
            "n_bootstrap": args.n_bootstrap,
            "n_permutations": args.n_permutations,
            "seed": args.seed,
            "force": args.force,
        })

    if tasks:
        workers = max(1, min(args.num_workers, len(tasks)))
        print(
            f"[INFO] Running significance with n_bootstrap={args.n_bootstrap}, "
            f"n_permutations={args.n_permutations}, workers={workers}"
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(compare_pair, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "ok":
                    print(f"[{result['detail'].upper()}] {result['model_a']} vs {result['model_b']}")
                    result_frames.append(pd.read_csv(result["out_csv"]))
                else:
                    print(f"[SKIP] {result['model_a']} vs {result['model_b']}: {result['reason']}")
                    skipped_rows.append({
                        "model_a": result["model_a"],
                        "model_b": result["model_b"],
                        "reason": result["reason"],
                        "detail": result["detail"],
                    })

    combined = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    skipped = pd.DataFrame(skipped_rows, columns=["model_a", "model_b", "reason", "detail"])

    combined.to_csv(out_dir / "pairwise_significance.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(out_dir / "skipped_pairs.csv", index=False, encoding="utf-8-sig")
    write_markdown(combined, skipped, out_dir / "pairwise_significance.md")

    print(f"[SUCCESS] Combined report: {out_dir / 'pairwise_significance.csv'}")
    print(f"[SUCCESS] Markdown report: {out_dir / 'pairwise_significance.md'}")
    if not skipped.empty:
        print(f"[INFO] Skipped pairs: {out_dir / 'skipped_pairs.csv'}")


if __name__ == "__main__":
    main()
