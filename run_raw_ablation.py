# run_raw_ablation.py
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable

# Базовые параметры для всех запусков
COMMON_ARGS = [
    PYTHON, "train.py",
    "--data_dir", "dataset",
    "--out_dir", "science_folder",
    "--epochs", "50",
    "--aug_type", "1",
    "--lr", "1e-4",
]

# Чтобы effective batch был сопоставим с твоими прошлыми экспериментами:
# 64 * 4 = 256 для обычных запусков
COMMON_STANDARD_BATCH = [
    "--batch_size", "64",
    "--accumulation_steps", "4",
]

# Для SupCon оставим тот же режим, что уже использовался ранее:
# 16 * 16 = 256
COMMON_SUPCON_BATCH = [
    "--batch_size", "16",
    "--accumulation_steps", "16",
]

EXPERIMENTS = [
    {
        "name": "ablation_raw_ce",
        "args": [
            "--exp_name", "ablation_raw_ce",
            "--loss", "ce",
            "--smoothing", "0.0",
        ] + COMMON_STANDARD_BATCH,
    },
    {
        "name": "ablation_raw_ce_weighted",
        "args": [
            "--exp_name", "ablation_raw_ce_weighted",
            "--loss", "ce",
            "--smoothing", "0.0",
            "--use_weights",
        ] + COMMON_STANDARD_BATCH,
    },
    {
        "name": "ablation_raw_focal_g1",
        "args": [
            "--exp_name", "ablation_raw_focal_g1",
            "--loss", "focal",
            "--gamma", "1.0",
        ] + COMMON_STANDARD_BATCH,
    },
    {
        "name": "ablation_raw_focal_g1_weighted",
        "args": [
            "--exp_name", "ablation_raw_focal_g1_weighted",
            "--loss", "focal",
            "--gamma", "1.0",
            "--use_weights",
        ] + COMMON_STANDARD_BATCH,
    },
    {
        "name": "ablation_raw_supcon",
        "args": [
            "--exp_name", "ablation_raw_supcon",
            "--loss", "ce",
            "--use_supcon",
            "--supcon_weight", "0.1",
        ] + COMMON_SUPCON_BATCH,
    },
]


def run_experiment(exp: dict):
    cmd = COMMON_ARGS + exp["args"]
    print("\n" + "=" * 80)
    print(f"[START] {exp['name']}")
    print(" ".join(cmd))
    print("=" * 80)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Эксперимент {exp['name']} завершился с ошибкой: {result.returncode}")

    print(f"[DONE] {exp['name']}")


def main():
    Path("science_folder").mkdir(parents=True, exist_ok=True)

    for exp in EXPERIMENTS:
        run_experiment(exp)

    print("\n[SUCCESS] Все RAW-ablation эксперименты завершены.")


if __name__ == "__main__":
    main()