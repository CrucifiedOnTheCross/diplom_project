#!/usr/bin/env python3
import argparse
import subprocess
import sys

PYTHON = sys.executable

COMMON = {
    "out_dir": "science_folder",
    "epochs": "50",
    "lr": "1e-4",
}

# ---- Эксперименты ----
EXPERIMENTS = {
    # Обязательные
    "21_raw_undersample_ce": {
        "exp_name": "21_raw_undersample_ce",
        "data_dir": "dataset",
        "aug_type": "1",
        "batch_size": "64",
        "accumulation_steps": "4",
        "loss": "ce",
        "sampling_mode": "undersample",
        "seed": "42",
    },
    "22_raw_oversample_ce": {
        "exp_name": "22_raw_oversample_ce",
        "data_dir": "dataset",
        "aug_type": "1",
        "batch_size": "64",
        "accumulation_steps": "4",
        "loss": "ce",
        "sampling_mode": "oversample",
        "seed": "42",
    },
    "23_raw_smooth_ce": {
        "exp_name": "23_raw_smooth_ce",
        "data_dir": "dataset",
        "aug_type": "1",
        "batch_size": "64",
        "accumulation_steps": "4",
        "loss": "ce",
        "smoothing": "0.1",
        "seed": "42",
    },
    "24_raw_supcon_weighted": {
        "exp_name": "24_raw_supcon_weighted",
        "data_dir": "dataset",
        "aug_type": "1",
        "batch_size": "16",
        "accumulation_steps": "16",
        "loss": "ce",
        "use_supcon": True,
        "supcon_weight": "0.1",
        "use_weights": True,
        "seed": "42",
    },
    "25_prep_focal_g1": {
        "exp_name": "25_prep_focal_g1",
        "data_dir": "dataset_preprocessed",
        "batch_size": "64",
        "accumulation_steps": "4",
        "loss": "focal",
        "gamma": "1.0",
        "seed": "42",
    },
    "26_gan_weighted_ce": {
        "exp_name": "26_gan_weighted_ce",
        "data_dir": "dataset_augmented",
        "batch_size": "64",
        "accumulation_steps": "4",
        "loss": "ce",
        "use_weights": True,
        "seed": "42",
    },
    "27_gan_focal_g1": {
        "exp_name": "27_gan_focal_g1",
        "data_dir": "dataset_augmented",
        "batch_size": "64",
        "accumulation_steps": "4",
        "loss": "focal",
        "gamma": "1.0",
        "seed": "42",
    },
    "28_gan_focal_g1_weighted": {
        "exp_name": "28_gan_focal_g1_weighted",
        "data_dir": "dataset_augmented",
        "batch_size": "64",
        "accumulation_steps": "4",
        "loss": "focal",
        "gamma": "1.0",
        "use_weights": True,
        "seed": "42",
    },

    # Желательные
    "29_raw_focal_g0.5": {
        "exp_name": "29_raw_focal_g0.5",
        "data_dir": "dataset",
        "aug_type": "1",
        "batch_size": "64",
        "accumulation_steps": "4",
        "loss": "focal",
        "gamma": "0.5",
        "seed": "42",
    },
    "30_raw_focal_g2.0": {
        "exp_name": "30_raw_focal_g2.0",
        "data_dir": "dataset",
        "aug_type": "1",
        "batch_size": "64",
        "accumulation_steps": "4",
        "loss": "focal",
        "gamma": "2.0",
        "seed": "42",
    },
    "31_raw_focal_g2.0_weighted": {
        "exp_name": "31_raw_focal_g2.0_weighted",
        "data_dir": "dataset",
        "aug_type": "1",
        "batch_size": "64",
        "accumulation_steps": "4",
        "loss": "focal",
        "gamma": "2.0",
        "use_weights": True,
        "seed": "42",
    },
    "32_prep_supcon_weighted": {
        "exp_name": "32_prep_supcon_weighted",
        "data_dir": "dataset_preprocessed",
        "batch_size": "16",
        "accumulation_steps": "16",
        "loss": "ce",
        "use_supcon": True,
        "supcon_weight": "0.1",
        "use_weights": True,
        "seed": "42",
    },
    "33_gan_supcon_weighted": {
        "exp_name": "33_gan_supcon_weighted",
        "data_dir": "dataset_augmented",
        "batch_size": "16",
        "accumulation_steps": "16",
        "loss": "ce",
        "use_supcon": True,
        "supcon_weight": "0.1",
        "use_weights": True,
        "seed": "42",
    },

    # Повторные прогоны
    "20_raw_supcon_seed42": {
        "exp_name": "20_raw_supcon_seed42",
        "data_dir": "dataset",
        "aug_type": "1",
        "batch_size": "16",
        "accumulation_steps": "16",
        "loss": "ce",
        "use_supcon": True,
        "supcon_weight": "0.1",
        "seed": "42",
    },
    "20_raw_supcon_seed52": {
        "exp_name": "20_raw_supcon_seed52",
        "data_dir": "dataset",
        "aug_type": "1",
        "batch_size": "16",
        "accumulation_steps": "16",
        "loss": "ce",
        "use_supcon": True,
        "supcon_weight": "0.1",
        "seed": "52",
    },
    "20_raw_supcon_seed62": {
        "exp_name": "20_raw_supcon_seed62",
        "data_dir": "dataset",
        "aug_type": "1",
        "batch_size": "16",
        "accumulation_steps": "16",
        "loss": "ce",
        "use_supcon": True,
        "supcon_weight": "0.1",
        "seed": "62",
    },

    "ablation_raw_focal_g1_weighted_seed42": {
        "exp_name": "ablation_raw_focal_g1_weighted_seed42",
        "data_dir": "dataset",
        "aug_type": "1",
        "batch_size": "64",
        "accumulation_steps": "4",
        "loss": "focal",
        "gamma": "1.0",
        "use_weights": True,
        "seed": "42",
    },
    "ablation_raw_focal_g1_weighted_seed52": {
        "exp_name": "ablation_raw_focal_g1_weighted_seed52",
        "data_dir": "dataset",
        "aug_type": "1",
        "batch_size": "64",
        "accumulation_steps": "4",
        "loss": "focal",
        "gamma": "1.0",
        "use_weights": True,
        "seed": "52",
    },
    "ablation_raw_focal_g1_weighted_seed62": {
        "exp_name": "ablation_raw_focal_g1_weighted_seed62",
        "data_dir": "dataset",
        "aug_type": "1",
        "batch_size": "64",
        "accumulation_steps": "4",
        "loss": "focal",
        "gamma": "1.0",
        "use_weights": True,
        "seed": "62",
    },

    "13_gan_supcon_seed42": {
        "exp_name": "13_gan_supcon_seed42",
        "data_dir": "dataset_augmented",
        "batch_size": "16",
        "accumulation_steps": "16",
        "loss": "ce",
        "use_supcon": True,
        "supcon_weight": "0.1",
        "seed": "42",
    },
    "13_gan_supcon_seed52": {
        "exp_name": "13_gan_supcon_seed52",
        "data_dir": "dataset_augmented",
        "batch_size": "16",
        "accumulation_steps": "16",
        "loss": "ce",
        "use_supcon": True,
        "supcon_weight": "0.1",
        "seed": "52",
    },
}

GROUPS = {
    "required": [
        "21_raw_undersample_ce",
        "22_raw_oversample_ce",
        "23_raw_smooth_ce",
        "24_raw_supcon_weighted",
        "25_prep_focal_g1",
        "26_gan_weighted_ce",
        "27_gan_focal_g1",
        "28_gan_focal_g1_weighted",
    ],
    "desired": [
        "29_raw_focal_g0.5",
        "30_raw_focal_g2.0",
        "31_raw_focal_g2.0_weighted",
        "32_prep_supcon_weighted",
        "33_gan_supcon_weighted",
    ],
    "repeats": [
        "20_raw_supcon_seed42",
        "20_raw_supcon_seed52",
        "20_raw_supcon_seed62",
        "ablation_raw_focal_g1_weighted_seed42",
        "ablation_raw_focal_g1_weighted_seed52",
        "ablation_raw_focal_g1_weighted_seed62",
        "13_gan_supcon_seed42",
        "13_gan_supcon_seed52",
    ],
}

def build_cmd(cfg):
    cmd = [PYTHON, "train.py"]
    full = dict(COMMON)
    full.update(cfg)

    for key, value in full.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["required", "desired", "repeats", "all"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Пропускать уже существующие папки экспериментов")
    args = parser.parse_args()

    names = []
    if args.group == "all":
        for g in ["required", "desired", "repeats"]:
            names.extend(GROUPS[g])
    else:
        names = GROUPS[args.group]

    for name in names:
        cfg = EXPERIMENTS[name]
        out_dir = full_out = f"{COMMON['out_dir']}/{cfg['exp_name']}"
        if args.skip_existing:
            import os
            if os.path.isdir(out_dir):
                print(f"[SKIP] {name} already exists")
                continue

        cmd = build_cmd(cfg)
        print("\n" + "=" * 100)
        print("[RUN]", name)
        print(" ".join(cmd))
        print("=" * 100)

        if not args.dry_run:
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
