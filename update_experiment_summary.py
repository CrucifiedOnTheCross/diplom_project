import os
import re
import csv
import json
from pathlib import Path

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


def parse_report(filepath):
    """Парсит medical_metrics_report.txt, заменяя все переносы строк на пробелы."""
    metrics = {}
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        raw_content = f.read()
        content = re.sub(r'\s+', ' ', raw_content)

    global_patterns = {
        'Accuracy': r'Standard Accuracy:\s+([0-9.]+)',
        'Balanced_Accuracy': r'Balanced Accuracy:\s+([0-9.]+)',
        'MCC': r'MCC \(Matthews Corr\):\s+([0-9.]+)',
        'PR_AUC': r'PR-AUC \(Macro\):\s+([0-9.]+)',
        'ROC_AUC_Macro': r'Macro Average ROC-AUC:\s+([0-9.]+)',
        'ROC_AUC_Weighted': r'Weighted Average ROC-AUC:\s+([0-9.]+)',
        'Sensitivity_Global': r'-\s+Sensitivity \(Recall\):\s+([0-9.]+)',
        'Specificity_Global': r'-\s+Specificity:\s+([0-9.]+)',
        'F1_Macro': r'macro avg\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)',
        'F1_Weighted': r'weighted avg\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)'
    }

    for key, pattern in global_patterns.items():
        match = re.search(pattern, content)
        metrics[key] = match.group(1) if match else "N/A"

    for cls in CLASSES:
        cls_pattern = rf'\b{cls}\b\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+[0-9]+'
        match = re.search(cls_pattern, content)
        if match:
            metrics[f'{cls}_Precision'] = match.group(1)
            metrics[f'{cls}_Recall'] = match.group(2)
            metrics[f'{cls}_F1'] = match.group(3)
        else:
            metrics[f'{cls}_Precision'] = "N/A"
            metrics[f'{cls}_Recall'] = "N/A"
            metrics[f'{cls}_F1'] = "N/A"

    for cls in CLASSES:
        spec_pattern = rf'-\s+{cls}\s*:\s*([0-9.]+)'
        match = re.search(spec_pattern, content)
        metrics[f'{cls}_Specificity'] = match.group(1) if match else "N/A"

    return metrics


def parse_calibration_report(filepath):
    calibration_metrics = {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        calibration_metrics["Calibration_Status"] = f"read_error: {e}"
        return calibration_metrics

    calibration_metrics["Calibration_Status"] = "OK"

    candidate_keys = [
        "temperature",
        "ece_before", "ece_after", "ece_delta",
        "brier_before", "brier_after", "brier_delta",
        "nll_before", "nll_after", "nll_delta",
    ]

    for key in candidate_keys:
        calibration_metrics[key] = data.get(key, "N/A")

    return calibration_metrics


def parse_threshold_report(filepath, prefix):
    """
    Читает threshold_report_*.json и возвращает плоский словарь с префиксом.
    prefix:
      - malignant
      - melanoma
    """
    out = {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        out[f"{prefix}_Threshold_Status"] = f"read_error: {e}"
        return out

    out[f"{prefix}_Threshold_Status"] = "OK"
    out[f"{prefix}_threshold_mode"] = data.get("mode", "N/A")
    out[f"{prefix}_threshold_criterion"] = data.get("criterion", "N/A")

    best_before = data.get("best_before", {}) or {}
    best_after = data.get("best_after", {}) or {}

    before_keys = [
        "threshold", "sensitivity", "specificity", "precision",
        "f1", "mcc", "balanced_accuracy", "accuracy", "youden"
    ]
    after_keys = before_keys

    for key in before_keys:
        out[f"{prefix}_{key}_before"] = best_before.get(key, "N/A")

    for key in after_keys:
        out[f"{prefix}_{key}_after"] = best_after.get(key, "N/A")

    def safe_delta(after_key, before_key, name):
        a = best_after.get(after_key, None)
        b = best_before.get(before_key, None)
        try:
            out[f"{prefix}_{name}_delta"] = float(a) - float(b)
        except Exception:
            out[f"{prefix}_{name}_delta"] = "N/A"

    safe_delta("threshold", "threshold", "threshold")
    safe_delta("sensitivity", "sensitivity", "sensitivity")
    safe_delta("specificity", "specificity", "specificity")
    safe_delta("precision", "precision", "precision")
    safe_delta("f1", "f1", "f1")
    safe_delta("mcc", "mcc", "mcc")
    safe_delta("balanced_accuracy", "balanced_accuracy", "balanced_accuracy")
    safe_delta("accuracy", "accuracy", "accuracy")
    safe_delta("youden", "youden", "youden")

    return out


def load_config(config_path):
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8', errors='ignore') as f:
            config_text = f.read()
        return config_text.replace('\n', ' ').replace('\r', ' ').strip()
    return "Config Missing"


def fill_missing_medical(exp_data):
    for key in [
        'Accuracy', 'Balanced_Accuracy', 'MCC', 'PR_AUC',
        'ROC_AUC_Macro', 'ROC_AUC_Weighted',
        'Sensitivity_Global', 'Specificity_Global',
        'F1_Macro', 'F1_Weighted'
    ]:
        exp_data[key] = "N/A"

    for cls in CLASSES:
        exp_data[f'{cls}_Precision'] = "N/A"
        exp_data[f'{cls}_Recall'] = "N/A"
        exp_data[f'{cls}_F1'] = "N/A"
        exp_data[f'{cls}_Specificity'] = "N/A"


def fill_missing_calibration(exp_data):
    exp_data["Calibration_Status"] = "NotComputed"
    for key in [
        "temperature",
        "ece_before", "ece_after", "ece_delta",
        "brier_before", "brier_after", "brier_delta",
        "nll_before", "nll_after", "nll_delta",
    ]:
        exp_data[key] = "N/A"


def fill_missing_threshold(exp_data, prefix):
    exp_data[f"{prefix}_Threshold_Status"] = "NotComputed"
    exp_data[f"{prefix}_threshold_mode"] = "N/A"
    exp_data[f"{prefix}_threshold_criterion"] = "N/A"

    keys = [
        "threshold", "sensitivity", "specificity", "precision",
        "f1", "mcc", "balanced_accuracy", "accuracy", "youden"
    ]

    for key in keys:
        exp_data[f"{prefix}_{key}_before"] = "N/A"
        exp_data[f"{prefix}_{key}_after"] = "N/A"
        exp_data[f"{prefix}_{key}_delta"] = "N/A"


def main():
    experiments_dir = Path('science_folder')
    if not experiments_dir.exists():
        print("Папка science_folder не найдена!")
        return

    all_results = []

    print("[*] Собираем данные экспериментов...")
    for exp_folder in sorted(experiments_dir.iterdir()):
        if not exp_folder.is_dir():
            continue

        test_report_path = exp_folder / 'medical_metrics_report_test.txt'
        valid_report_path = exp_folder / 'medical_metrics_report.txt'
        report_path = test_report_path if test_report_path.exists() else valid_report_path
        config_path = exp_folder / 'config.txt'
        calibration_path = exp_folder / 'calibration_report.json'

        malignant_threshold_path = exp_folder / 'threshold_report_malignant_youden.json'
        melanoma_threshold_path = exp_folder / 'threshold_report_melanoma_youden.json'

        has_medical = report_path.exists()
        has_calibration = calibration_path.exists()
        has_thr_malignant = malignant_threshold_path.exists()
        has_thr_melanoma = melanoma_threshold_path.exists()

        if not has_medical and not has_calibration and not has_thr_malignant and not has_thr_melanoma:
            continue

        exp_data = {'Experiment_Folder': exp_folder.name}
        exp_data['Config'] = load_config(config_path)
        exp_data['Medical_Report_Source'] = 'test' if test_report_path.exists() else 'valid_legacy'

        if has_medical:
            exp_data.update(parse_report(report_path))
        else:
            fill_missing_medical(exp_data)

        if has_calibration:
            exp_data.update(parse_calibration_report(calibration_path))
        else:
            fill_missing_calibration(exp_data)

        if has_thr_malignant:
            exp_data.update(parse_threshold_report(malignant_threshold_path, "malignant"))
        else:
            fill_missing_threshold(exp_data, "malignant")

        if has_thr_melanoma:
            exp_data.update(parse_threshold_report(melanoma_threshold_path, "melanoma"))
        else:
            fill_missing_threshold(exp_data, "melanoma")

        all_results.append(exp_data)

        status_parts = []
        status_parts.append("medical=YES" if has_medical else "medical=NO")
        status_parts.append("calibration=YES" if has_calibration else "calibration=NO")
        status_parts.append("thr_malignant=YES" if has_thr_malignant else "thr_malignant=NO")
        status_parts.append("thr_melanoma=YES" if has_thr_melanoma else "thr_melanoma=NO")
        print(f"  [+] {exp_folder.name} | " + " | ".join(status_parts))

    if not all_results:
        print("\n[!] Нет отчетов для сбора.")
        return

    output_file = 'all_experiments_summary.csv'

    base_fields = [
        'Experiment_Folder', 'Config', 'Medical_Report_Source',
        'Balanced_Accuracy', 'MCC',
        'F1_Macro', 'F1_Weighted',
        'Accuracy', 'Sensitivity_Global', 'Specificity_Global',
        'PR_AUC', 'ROC_AUC_Macro', 'ROC_AUC_Weighted'
    ]

    calibration_fields = [
        "Calibration_Status",
        "temperature",
        "ece_before", "ece_after", "ece_delta",
        "brier_before", "brier_after", "brier_delta",
        "nll_before", "nll_after", "nll_delta",
    ]

    threshold_metric_names = [
        "threshold", "sensitivity", "specificity", "precision",
        "f1", "mcc", "balanced_accuracy", "accuracy", "youden"
    ]

    threshold_fields = []

    for prefix in ["malignant", "melanoma"]:
        threshold_fields.extend([
            f"{prefix}_Threshold_Status",
            f"{prefix}_threshold_mode",
            f"{prefix}_threshold_criterion",
        ])
        for key in threshold_metric_names:
            threshold_fields.append(f"{prefix}_{key}_before")
        for key in threshold_metric_names:
            threshold_fields.append(f"{prefix}_{key}_after")
        for key in threshold_metric_names:
            threshold_fields.append(f"{prefix}_{key}_delta")

    class_fields = []
    for cls in CLASSES:
        class_fields.extend([
            f'{cls}_Recall',
            f'{cls}_Specificity',
            f'{cls}_Precision',
            f'{cls}_F1'
        ])

    fieldnames = base_fields + calibration_fields + threshold_fields + class_fields

    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()

        for row in all_results:
            clean_row = {}
            for field in fieldnames:
                val = str(row.get(field, 'N/A'))
                clean_row[field] = val.replace('\n', ' ').replace('\r', ' ')
            writer.writerow(clean_row)

    print(f"\n[SUCCESS] Таблица успешно сохранена в {output_file}")


if __name__ == '__main__':
    main()
