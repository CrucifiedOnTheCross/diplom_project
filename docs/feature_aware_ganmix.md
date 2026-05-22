# Feature-aware GAN-mix

## Goal

This add-on experiment tests SupCon-guided GAN-mix without training new generators.
It reuses already generated StyleGAN2-ADA synthetic images and selects them by their
position in the embedding space of a SupCon or weighted SupCon ConvNeXt model.

The core hypothesis is simple: synthetic images that sit inside the real class core
and are not closer to another class centroid should be safer augmentation candidates
than arbitrary synthetic samples.

## Added scripts

- `export_embeddings.py` exports `embeddings.npy`, `labels.npy`, `image_paths.txt`, and `metadata.csv`.
- `analyze_embedding_space.py` computes class centroids, within-class distances, centroid distances, silhouette, and 2D plots.
- `build_feature_aware_ganmix_manifest.py` selects synthetic images with `random`, `core`, or `diverse_core`.
- `build_manifest_dataset.py` clones an existing ImageFolder dataset and adds selected synthetic images into `train/<class>`.
- `make_feature_aware_report.py` builds comparison tables and figures from `all_experiments_summary.csv`.

## Inputs

Expected existing inputs:

- a SupCon or weighted SupCon checkpoint, for example `science_folder/24_raw_supcon_weighted/best_model.pth`;
- a base classifier dataset with `train/valid/test`, for example `dataset` or `dataset_preprocessed_v2`;
- existing synthetic images, for example under `synthetic_mixing_runs`;
- either ImageFolder-style synthetic directories or CSV files with columns `image_path,label`.

No new GAN training is required.

## Server workflow

Set paths once:

```bash
SUPCON_CKPT="science_folder/24_raw_supcon_weighted/best_model.pth"
BASE_DATASET="dataset"
EMB_ROOT="embedding_analysis/24_raw_supcon_weighted"
SYNTH_CSV="synthetic_mixing_runs/all_synthetic.csv"
```

If synthetic images are already organized as one ImageFolder directory with class
subfolders, use `--image-dir` instead of `--split-csv` in the synthetic export step.

Export real train embeddings:

```bash
python export_embeddings.py \
  --checkpoint "$SUPCON_CKPT" \
  --image-dir "$BASE_DATASET/train" \
  --split train \
  --source real \
  --output-dir "$EMB_ROOT/train" \
  --batch-size 64 \
  --device cuda
```

Export validation/test embeddings for analysis:

```bash
python export_embeddings.py \
  --checkpoint "$SUPCON_CKPT" \
  --image-dir "$BASE_DATASET/test" \
  --split test \
  --source real \
  --output-dir "$EMB_ROOT/test" \
  --batch-size 64 \
  --device cuda
```

Export synthetic embeddings:

```bash
python export_embeddings.py \
  --checkpoint "$SUPCON_CKPT" \
  --split-csv "$SYNTH_CSV" \
  --source synthetic \
  --output-dir "$EMB_ROOT/synthetic" \
  --batch-size 64 \
  --device cuda
```

Analyze embedding space:

```bash
python analyze_embedding_space.py \
  --embeddings "$EMB_ROOT/test/embeddings.npy" \
  --labels "$EMB_ROOT/test/labels.npy" \
  --metadata "$EMB_ROOT/test/metadata.csv" \
  --output-dir "$EMB_ROOT/reports"
```

Build manifests:

```bash
python build_feature_aware_ganmix_manifest.py \
  --real-train-embeddings "$EMB_ROOT/train/embeddings.npy" \
  --real-train-metadata "$EMB_ROOT/train/metadata.csv" \
  --synthetic-embeddings "$EMB_ROOT/synthetic/embeddings.npy" \
  --synthetic-metadata "$EMB_ROOT/synthetic/metadata.csv" \
  --selection-mode random \
  --synthetic-ratio 0.25 \
  --output synthetic_mixing_runs/random_ganmix_25/manifest.csv

python build_feature_aware_ganmix_manifest.py \
  --real-train-embeddings "$EMB_ROOT/train/embeddings.npy" \
  --real-train-metadata "$EMB_ROOT/train/metadata.csv" \
  --synthetic-embeddings "$EMB_ROOT/synthetic/embeddings.npy" \
  --synthetic-metadata "$EMB_ROOT/synthetic/metadata.csv" \
  --selection-mode diverse_core \
  --synthetic-ratio 0.25 \
  --output synthetic_mixing_runs/feature_aware_ganmix_25/manifest.csv
```

Build ImageFolder datasets from manifests:

```bash
python build_manifest_dataset.py \
  --src-dataset "$BASE_DATASET" \
  --manifest synthetic_mixing_runs/random_ganmix_25/manifest.csv \
  --out-dataset datasets_feature_aware/random_ganmix_25 \
  --link-mode hardlink

python build_manifest_dataset.py \
  --src-dataset "$BASE_DATASET" \
  --manifest synthetic_mixing_runs/feature_aware_ganmix_25/manifest.csv \
  --out-dataset datasets_feature_aware/diverse_core_ganmix_25 \
  --link-mode hardlink
```

Run the three short classifier experiments on the server:

```bash
python train.py \
  --out_dir science_folder \
  --exp_name random_ganmix_25_raw_ce \
  --data_dir datasets_feature_aware/random_ganmix_25 \
  --aug_type 1 \
  --batch_size 16 \
  --accumulation_steps 16 \
  --epochs 50 \
  --lr 1e-4 \
  --loss ce \
  --seed 42

python train.py \
  --out_dir science_folder \
  --exp_name diverse_core_ganmix_25_raw_ce \
  --data_dir datasets_feature_aware/diverse_core_ganmix_25 \
  --aug_type 1 \
  --batch_size 16 \
  --accumulation_steps 16 \
  --epochs 50 \
  --lr 1e-4 \
  --loss ce \
  --seed 42

python train.py \
  --out_dir science_folder \
  --exp_name diverse_core_ganmix_25_weighted_ce \
  --data_dir datasets_feature_aware/diverse_core_ganmix_25 \
  --aug_type 1 \
  --batch_size 16 \
  --accumulation_steps 16 \
  --epochs 50 \
  --lr 1e-4 \
  --loss ce \
  --use_weights \
  --seed 42
```

After training, reuse the existing evaluation pipeline:

```bash
python calibration_eval.py --experiments_root science_folder
python threshold_eval.py --mode malignant --criterion youden
python threshold_eval.py --mode melanoma --criterion youden
python update_experiment_summary.py
```

Build the feature-aware report:

```bash
python make_feature_aware_report.py \
  --summary all_experiments_summary.csv \
  --baseline baseline_real_only_raw_ganmix \
  --experiments random_ganmix_25_raw_ce diverse_core_ganmix_25_raw_ce diverse_core_ganmix_25_weighted_ce \
  --output-dir feature_aware_report
```

Change `--baseline` to the actual baseline folder name in `science_folder` if it differs.

## Outputs

Embedding export:

- `embeddings.npy`
- `labels.npy`
- `image_paths.txt`
- `metadata.csv`

Embedding analysis:

- `class_centroids.npy`
- `class_centroid_labels.txt`
- `class_distance_stats.csv`
- `centroid_distances.csv`
- `silhouette_summary.csv`
- `focus_class_pairs.csv`
- `nearest_conflicting_classes.csv`
- `umap_by_true_label.png`, `tsne_by_true_label.png`, or `pca_by_true_label.png`
- matching `*_errors_highlighted.png` when predictions exist

GAN-mix selection:

- `manifest.csv`
- `manifest_summary.csv`

Final report:

- `feature_aware_summary.csv`
- `feature_aware_delta_summary.csv`
- `figures/*.png`

## Do not commit

Do not commit generated datasets, embeddings, synthetic images, model weights, or reports:

- `dataset/`, `dataset_*`, `datasets_*`
- `embedding_analysis/`
- `synthetic_mixing_runs/`
- `science_folder/`
- `diploma_results*`
- `diploma_figures/`
- `gradcam_results/`, `gradcam_diploma/`
- `threshold_analysis*`, `binary_threshold_analysis*`
- `*.pth`, `*.pt`, `*.ckpt`, `*.zip`, `*.tar.gz`
- `.env`, `.env.*`
