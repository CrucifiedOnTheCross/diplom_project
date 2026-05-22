from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from model import JointSkinLesionClassifier

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
DEFAULT_BATCH_SIZE = 64
FINAL_MARKERS = [
    'medical_metrics_report.txt',
    'training_results.png',
]


# =========================
# Utility
# =========================
def clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            cleaned[k.replace('_orig_mod.', '')] = v
        else:
            cleaned[k] = v
    return cleaned


def parse_config(config_path: Path) -> Dict[str, str]:
    """
    config.txt имеет формат:
    key: value
    """
    cfg: Dict[str, str] = {}
    if not config_path.exists():
        return cfg

    with open(config_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            k, v = line.split(':', 1)
            cfg[k.strip()] = v.strip()
    return cfg


def infer_eval_dirs(data_dir: str) -> Tuple[Path, Path, str]:
    """
    Возвращает valid_dir, test_dir и строку-описание политики.

    Логика:
    - dataset / dataset_preprocessed / dataset_preprocessed_v2: берём их собственные valid/test
    - dataset_augmented: обычно содержит только train, поэтому valid/test берём из dataset_preprocessed
    """
    root = Path(data_dir)

    valid_dir = root / 'valid'
    test_dir = root / 'test'
    if valid_dir.exists() and test_dir.exists():
        return valid_dir, test_dir, f'direct:{root}'

    if root.name == 'dataset_augmented':
        fallback = Path('dataset_preprocessed')
        valid_dir = fallback / 'valid'
        test_dir = fallback / 'test'
        return valid_dir, test_dir, 'fallback:dataset_augmented->dataset_preprocessed'

    return valid_dir, test_dir, 'missing'


def has_final_markers(exp_dir: Path) -> bool:
    return any((exp_dir / marker).exists() for marker in FINAL_MARKERS)


def is_checkpoint_stable(model_path: Path, min_age_sec: int) -> Tuple[bool, str]:
    if not model_path.exists():
        return False, 'best_model.pth not found'

    try:
        age_sec = time.time() - model_path.stat().st_mtime
    except FileNotFoundError:
        return False, 'best_model.pth disappeared during stat'

    if age_sec < min_age_sec:
        return False, f'checkpoint is too recent ({age_sec:.1f}s < {min_age_sec}s)'

    return True, f'checkpoint age {age_sec:.1f}s'


def get_experiment_status(
    exp_dir: Path,
    force: bool,
    checkpoint_min_age_sec: int,
) -> Tuple[str, str]:
    """
    Возвращает (status, reason), где status ∈ {'process', 'skip'}.
    """
    model_path = exp_dir / 'best_model.pth'
    report_json_path = exp_dir / 'calibration_report.json'
    bins_csv_path = exp_dir / 'calibration_bins.csv'

    if report_json_path.exists() and bins_csv_path.exists() and not force:
        return 'skip', 'calibration already exists'

    if not model_path.exists():
        return 'skip', 'best_model.pth not found'

    if not has_final_markers(exp_dir):
        return 'skip', 'experiment looks active: final artifacts are missing'

    stable, reason = is_checkpoint_stable(model_path, checkpoint_min_age_sec)
    if not stable:
        return 'skip', f'experiment looks active: {reason}'

    return 'process', reason


def append_skip_log(skip_log_path: Path, experiment: str, reason: str) -> None:
    file_exists = skip_log_path.exists()
    with open(skip_log_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['timestamp', 'experiment', 'reason'],
            delimiter=';'
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'experiment': experiment,
            'reason': reason,
        })


# =========================
# Data / Model
# =========================
def build_loader(data_dir: Path, batch_size: int = DEFAULT_BATCH_SIZE) -> Tuple[datasets.ImageFolder, DataLoader]:
    transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(str(data_dir), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return dataset, loader


def load_model(model_path: Path, device: torch.device) -> nn.Module:
    model = JointSkinLesionClassifier(num_classes=len(CLASS_NAMES)).to(device)
    state = torch.load(model_path, map_location=device)
    state = clean_state_dict(state)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def collect_logits_and_labels(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    logits_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images, return_embeddings=False)
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(labels.numpy())

    logits_np = np.concatenate(logits_all, axis=0)
    labels_np = np.concatenate(labels_all, axis=0)
    return logits_np, labels_np


# =========================
# Calibration metrics
# =========================
def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def multiclass_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    one_hot = np.eye(num_classes, dtype=np.float64)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Top-label ECE для multiclass.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correctness = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_rows: List[Dict[str, float]] = []

    for i in range(n_bins):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        count = int(mask.sum())
        if count == 0:
            bin_rows.append({
                'bin_id': i,
                'lower': lo,
                'upper': hi,
                'count': 0,
                'avg_confidence': math.nan,
                'avg_accuracy': math.nan,
                'gap': math.nan,
            })
            continue

        avg_conf = float(confidences[mask].mean())
        avg_acc = float(correctness[mask].mean())
        gap = abs(avg_acc - avg_conf)
        ece += (count / len(labels)) * gap

        bin_rows.append({
            'bin_id': i,
            'lower': lo,
            'upper': hi,
            'count': count,
            'avg_confidence': avg_conf,
            'avg_accuracy': avg_acc,
            'gap': gap,
        })

    return float(ece), bin_rows


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp_min(1e-6)


def fit_temperature_scaling(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    max_iter: int = 100,
) -> float:
    scaler = TemperatureScaler().to(device)
    criterion = nn.CrossEntropyLoss()

    logits_t = torch.tensor(val_logits, dtype=torch.float32, device=device)
    labels_t = torch.tensor(val_labels, dtype=torch.long, device=device)

    optimizer = torch.optim.LBFGS(
        scaler.parameters(),
        lr=0.05,
        max_iter=max_iter,
        line_search_fn='strong_wolfe',
    )

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits_t), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(scaler.temperature.detach().cpu().item())


def evaluate_probabilities(
    logits: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    n_bins: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    probs = softmax_np(logits)
    ece, bins = expected_calibration_error(probs, labels, n_bins=n_bins)
    brier = multiclass_brier_score(probs, labels, num_classes=num_classes)
    nll = float(log_loss(labels, probs, labels=list(range(num_classes))))

    metrics = {
        'ECE': ece,
        'Brier': brier,
        'NLL': nll,
    }
    return metrics, bins


# =========================
# Per-experiment processing
# =========================
def evaluate_experiment(
    exp_dir: Path,
    device: torch.device,
    batch_size: int,
    n_bins: int,
) -> Dict[str, object]:
    model_path = exp_dir / 'best_model.pth'
    config_path = exp_dir / 'config.txt'
    report_json_path = exp_dir / 'calibration_report.json'
    bins_csv_path = exp_dir / 'calibration_bins.csv'

    cfg = parse_config(config_path)
    data_dir = cfg.get('data_dir', 'dataset')
    valid_dir, test_dir, eval_policy = infer_eval_dirs(data_dir)

    if not valid_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f'valid/test dirs not found ({valid_dir}, {test_dir})')

    print(f'\n[START] {exp_dir.name}')
    print(f'[*] data_dir from config: {data_dir}')
    print(f'[*] eval policy: {eval_policy}')
    print(f'[*] valid: {valid_dir}')
    print(f'[*] test : {test_dir}')

    valid_dataset, valid_loader = build_loader(valid_dir, batch_size=batch_size)
    test_dataset, test_loader = build_loader(test_dir, batch_size=batch_size)

    if valid_dataset.classes != CLASS_NAMES:
        print(f'[WARN] {exp_dir.name}: unexpected valid class order: {valid_dataset.classes}')
    if test_dataset.classes != CLASS_NAMES:
        print(f'[WARN] {exp_dir.name}: unexpected test class order: {test_dataset.classes}')

    model = load_model(model_path, device=device)

    val_logits, val_labels = collect_logits_and_labels(model, valid_loader, device)
    test_logits, test_labels = collect_logits_and_labels(model, test_loader, device)

    pre_metrics, pre_bins = evaluate_probabilities(
        test_logits, test_labels, num_classes=len(CLASS_NAMES), n_bins=n_bins
    )

    temperature = fit_temperature_scaling(val_logits, val_labels, device=device)
    test_logits_ts = test_logits / max(temperature, 1e-6)
    post_metrics, post_bins = evaluate_probabilities(
        test_logits_ts, test_labels, num_classes=len(CLASS_NAMES), n_bins=n_bins
    )

    result = {
        'experiment': exp_dir.name,
        'model_path': str(model_path),
        'data_dir': data_dir,
        'eval_policy': eval_policy,
        'valid_dir': str(valid_dir),
        'test_dir': str(test_dir),
        'num_valid': int(len(valid_dataset)),
        'num_test': int(len(test_dataset)),
        'temperature': float(temperature),
        'ece_before': float(pre_metrics['ECE']),
        'brier_before': float(pre_metrics['Brier']),
        'nll_before': float(pre_metrics['NLL']),
        'ece_after': float(post_metrics['ECE']),
        'brier_after': float(post_metrics['Brier']),
        'nll_after': float(post_metrics['NLL']),
        'ece_delta': float(post_metrics['ECE'] - pre_metrics['ECE']),
        'brier_delta': float(post_metrics['Brier'] - pre_metrics['Brier']),
        'nll_delta': float(post_metrics['NLL'] - pre_metrics['NLL']),
        'n_bins': int(n_bins),
        'bins_before': pre_bins,
        'bins_after': post_bins,
    }

    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(bins_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['stage', 'bin_id', 'lower', 'upper', 'count', 'avg_confidence', 'avg_accuracy', 'gap']
        )
        writer.writeheader()
        for row in pre_bins:
            writer.writerow({'stage': 'before', **row})
        for row in post_bins:
            writer.writerow({'stage': 'after', **row})

    print(
        f"[DONE] {exp_dir.name} | "
        f"T={temperature:.4f} | "
        f"ECE {pre_metrics['ECE']:.4f}->{post_metrics['ECE']:.4f} | "
        f"Brier {pre_metrics['Brier']:.4f}->{post_metrics['Brier']:.4f} | "
        f"NLL {pre_metrics['NLL']:.4f}->{post_metrics['NLL']:.4f}"
    )
    return result


# =========================
# Summary aggregation
# =========================
def write_summary(results: List[Dict[str, object]], out_csv: Path) -> None:
    rows = []
    for r in results:
        if r is None:
            continue
        rows.append({
            'experiment': r['experiment'],
            'data_dir': r['data_dir'],
            'eval_policy': r['eval_policy'],
            'num_valid': r['num_valid'],
            'num_test': r['num_test'],
            'temperature': r['temperature'],
            'ece_before': r['ece_before'],
            'ece_after': r['ece_after'],
            'ece_delta': r['ece_delta'],
            'brier_before': r['brier_before'],
            'brier_after': r['brier_after'],
            'brier_delta': r['brier_delta'],
            'nll_before': r['nll_before'],
            'nll_after': r['nll_after'],
            'nll_delta': r['nll_delta'],
        })

    rows.sort(key=lambda x: x['experiment'])

    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'experiment', 'data_dir', 'eval_policy', 'num_valid', 'num_test',
                'temperature',
                'ece_before', 'ece_after', 'ece_delta',
                'brier_before', 'brier_after', 'brier_delta',
                'nll_before', 'nll_after', 'nll_delta',
            ],
            delimiter=';'
        )
        writer.writeheader()
        writer.writerows(rows)


# =========================
# Main
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Calibration-aware evaluation for all experiment folders.'
    )
    parser.add_argument('--experiments_root', type=str, default='science_folder')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--bins', type=int, default=15)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--force', action='store_true', help='Recompute even if calibration files already exist')
    parser.add_argument(
        '--checkpoint_min_age_sec',
        type=int,
        default=120,
        help='Skip experiments whose best_model.pth is newer than this threshold',
    )
    parser.add_argument(
        '--skip_log_name',
        type=str,
        default='calibration_skip_log.csv',
        help='CSV log filename for skipped experiments',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    root = Path(args.experiments_root)

    if not root.exists():
        raise FileNotFoundError(f'Экспериментальная папка не найдена: {root}')

    exp_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not exp_dirs:
        print(f'В папке {root} не найдено ни одной директории эксперимента.')
        return

    skip_log_path = root / args.skip_log_name
    run_skip_records: List[Dict[str, str]] = []
    results: List[Dict[str, object]] = []

    for exp_dir in exp_dirs:
        status, reason = get_experiment_status(
            exp_dir=exp_dir,
            force=args.force,
            checkpoint_min_age_sec=args.checkpoint_min_age_sec,
        )

        if status == 'skip':
            print(f'[SKIP] {exp_dir.name}: {reason}')
            append_skip_log(skip_log_path, exp_dir.name, reason)
            run_skip_records.append({'experiment': exp_dir.name, 'reason': reason})
            continue

        try:
            result = evaluate_experiment(
                exp_dir=exp_dir,
                device=device,
                batch_size=args.batch_size,
                n_bins=args.bins,
            )
            results.append(result)
        except Exception as e:
            reason = f'processing error: {e}'
            print(f'[ERROR] {exp_dir.name}: {e}')
            append_skip_log(skip_log_path, exp_dir.name, reason)
            run_skip_records.append({'experiment': exp_dir.name, 'reason': reason})

    summary_csv = root / 'calibration_summary.csv'
    write_summary(results, summary_csv)
    print(f'\n[SUCCESS] Summary saved to: {summary_csv.resolve()}')

    if run_skip_records:
        print('\n[SKIP SUMMARY]')
        for rec in run_skip_records:
            print(f" - {rec['experiment']}: {rec['reason']}")
        print(f'[*] Skip log appended to: {skip_log_path.resolve()}')
    else:
        print('\n[*] No experiments were skipped in this run.')


if __name__ == '__main__':
    main()
