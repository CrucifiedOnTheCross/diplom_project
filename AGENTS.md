# Project instructions for Codex

## Project context

This is a Python ML research project for a bachelor diploma about skin lesion classification on HAM10000 with class imbalance handling, StyleGAN-based augmentation, ConvNeXt classification, SupCon, calibration, threshold analysis, Grad-CAM, and experiment reporting.

The project contains source code and experiment scripts only. Large datasets, generated images, model weights, outputs, logs, figures, and external repositories must not be committed.

## Repository rules

Keep in Git:
- Python source files: `*.py`
- Shell scripts: `*.sh`
- Project documentation: `README.md`, `AGENTS.md`
- Git/service files: `.gitignore`
- Lightweight configs, if added later

Do not add to Git:
- `dataset/`, `dataset_*`, `datasets_*`
- `gan_data*`
- `gan_training_runs*`
- `science_folder/`
- `diploma_results*`
- `diploma_figures/`
- `gradcam_results/`, `gradcam_diploma/`
- `threshold_analysis*`, `binary_threshold_analysis*`
- `synthetic_mixing_runs/`
- `stylegan3-brecahad/`
- `*.pth`, `*.pt`, `*.ckpt`, `*.tar.gz`, `*.zip`
- `.env`, `.env.*`
- `audit_md*`

## Safety rules

Before changing code:
1. Inspect relevant files.
2. Explain the intended change briefly.
3. Prefer minimal, reversible changes.
4. Do not delete data, outputs, weights, or experiment folders.
5. Do not rewrite Git history unless explicitly requested.
6. Do not run long training jobs unless explicitly requested.
7. For remote commands, prefer scripts in `scripts/` and show the exact command before running it.

## Validation commands

For quick local validation:
```bash
python -m py_compile *.py