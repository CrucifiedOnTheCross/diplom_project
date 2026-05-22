#!/usr/bin/env python3
"""Run the post-experiment reporting pipeline.

This script does not train models. It orchestrates existing evaluation/report
scripts after experiments in science_folder have finished.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Step:
    name: str
    cmd: List[str]
    required_inputs: List[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all available post-experiment reports.")
    parser.add_argument("--science-dir", default="science_folder")
    parser.add_argument("--summary", default="all_experiments_summary.csv")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--threshold-criterion", default="youden", choices=["youden", "mcc", "f1", "balanced_accuracy"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--strict", action="store_true", help="Stop on the first failed optional step")
    parser.add_argument("--force", action="store_true", help="Forward --force to calibration/threshold scripts")

    parser.add_argument("--include-feature-aware", action="store_true")
    parser.add_argument("--feature-aware-baseline", default="")
    parser.add_argument(
        "--feature-aware-experiments",
        nargs="*",
        default=[
            "random_ganmix_25_raw_ce",
            "diverse_core_ganmix_25_raw_ce",
            "diverse_core_ganmix_25_weighted_ce",
        ],
    )
    parser.add_argument("--feature-aware-output", default="feature_aware_report")

    parser.add_argument("--include-gradcam", action="store_true")
    parser.add_argument("--include-diploma-figures", action="store_true")
    parser.add_argument("--diploma-fig-dir", default="diploma_figures")
    parser.add_argument("--diploma-model-path", default="science_folder/20_raw_supcon/best_model.pth")
    parser.add_argument("--diploma-dataset-root", default="dataset")

    parser.add_argument("--log-dir", default="report_runs")
    return parser.parse_args()


def script(name: str) -> str:
    return str(Path(name))


def existing_scripts(names: List[str]) -> List[Path]:
    return [Path(name) for name in names if Path(name).exists()]


def build_steps(args: argparse.Namespace) -> List[Step]:
    py = sys.executable
    science_dir = Path(args.science_dir)
    summary = Path(args.summary)

    calibration_cmd = [
        py,
        script("calibration_eval.py"),
        "--experiments_root",
        args.science_dir,
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
    ]
    if args.force:
        calibration_cmd.append("--force")

    threshold_base = [
        py,
        script("threshold_eval.py"),
        "--experiments_dir",
        args.science_dir,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--criterion",
        args.threshold_criterion,
    ]
    if args.force:
        threshold_base.append("--force")

    steps = [
        Step("calibration", calibration_cmd, [science_dir, Path("calibration_eval.py")]),
        Step("threshold_malignant", threshold_base + ["--mode", "malignant"], [science_dir, Path("threshold_eval.py")]),
        Step("threshold_melanoma", threshold_base + ["--mode", "melanoma"], [science_dir, Path("threshold_eval.py")]),
        Step("experiment_summary", [py, script("update_experiment_summary.py")], [science_dir, Path("update_experiment_summary.py")]),
    ]

    for plot_script in existing_scripts(["plot_pareto.py", "plot_clinical_pareto.py", "plot_threshold_shift.py"]):
        steps.append(Step(plot_script.stem, [py, str(plot_script)], [summary, plot_script]))

    if args.include_feature_aware:
        if not args.feature_aware_baseline:
            raise ValueError("--feature-aware-baseline is required with --include-feature-aware")
        steps.append(
            Step(
                "feature_aware_report",
                [
                    py,
                    script("make_feature_aware_report.py"),
                    "--summary",
                    args.summary,
                    "--baseline",
                    args.feature_aware_baseline,
                    "--experiments",
                    *args.feature_aware_experiments,
                    "--output-dir",
                    args.feature_aware_output,
                ],
                [summary, Path("make_feature_aware_report.py")],
            )
        )

    if args.include_gradcam:
        steps.append(Step("gradcam", [py, script("gradcam_grid_by_class.py")], [Path("gradcam_grid_by_class.py")]))

    if args.include_diploma_figures:
        steps.append(
            Step(
                "diploma_figures",
                [
                    py,
                    script("fix_diploma_figures_v2.py"),
                    "--fig-dir",
                    args.diploma_fig_dir,
                    "--dataset-root",
                    args.diploma_dataset_root,
                    "--science-dir",
                    args.science_dir,
                    "--model-path",
                    args.diploma_model_path,
                ],
                [summary, Path("fix_diploma_figures_v2.py"), Path(args.diploma_model_path)],
            )
        )

    return steps


def missing_inputs(step: Step) -> List[Path]:
    return [path for path in step.required_inputs if not path.exists()]


def run_step(step: Step) -> dict:
    start = time.time()
    print("\n" + "=" * 100)
    print(f"[RUN] {step.name}")
    print(" ".join(step.cmd))
    print("=" * 100)

    result = subprocess.run(step.cmd)
    elapsed = time.time() - start
    status = "ok" if result.returncode == 0 else "failed"
    print(f"[{status.upper()}] {step.name} in {elapsed:.1f}s")
    return {
        "step": step.name,
        "status": status,
        "returncode": result.returncode,
        "elapsed_sec": f"{elapsed:.1f}",
        "command": " ".join(step.cmd),
    }


def write_run_log(log_dir: Path, rows: List[dict]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / time.strftime("report_run_%Y%m%d_%H%M%S.csv")
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "status", "returncode", "elapsed_sec", "command"])
        writer.writeheader()
        writer.writerows(rows)
    return out


def main() -> None:
    args = parse_args()
    steps = build_steps(args)
    rows: List[dict] = []

    for step in steps:
        missing = missing_inputs(step)
        if missing:
            row = {
                "step": step.name,
                "status": "skipped",
                "returncode": "",
                "elapsed_sec": "0.0",
                "command": f"missing inputs: {', '.join(str(p) for p in missing)}",
            }
            rows.append(row)
            print(f"[SKIP] {step.name}: {row['command']}")
            if args.strict:
                break
            continue

        row = run_step(step)
        rows.append(row)
        if row["status"] != "ok" and args.strict:
            break

    log_path = write_run_log(Path(args.log_dir), rows)
    failed = [row for row in rows if row["status"] == "failed"]
    skipped = [row for row in rows if row["status"] == "skipped"]

    print("\n" + "=" * 100)
    print("[SUMMARY]")
    print(f"Log: {log_path.resolve()}")
    print(f"OK: {sum(row['status'] == 'ok' for row in rows)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed: {len(failed)}")
    print("=" * 100)

    if failed and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
