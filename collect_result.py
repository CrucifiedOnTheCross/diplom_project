import os
import re
import csv
from pathlib import Path

def parse_report(filepath):
    """Парсит текстовый файл отчета с защитой от 'битых' пробелов и кодировок."""
    metrics = {}
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        # МАГИЯ ЗДЕСЬ: превращаем любые странные невидимые пробелы (включая \xa0) в обычные пробелы
        raw_content = f.read()
        content = re.sub(r'[^\S\r\n]+', ' ', raw_content)

    # 1. Глобальные метрики
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

    # 2. Поклассовые метрики: Precision, Recall, F1
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    for cls in classes:
        # \b гарантирует, что мы ищем точное слово класса
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

    # 3. Поклассовые метрики: Specificity
    for cls in classes:
        spec_pattern = rf'-\s+{cls}\s*:\s*([0-9.]+)'
        match = re.search(spec_pattern, content)
        metrics[f'{cls}_Specificity'] = match.group(1) if match else "N/A"

    return metrics

def main():
    experiments_dir = Path('diploma_results__full')
    if not experiments_dir.exists():
        print("Папка experiments не найдена!")
        return

    all_results = []

    print("[*] Собираем данные экспериментов...")
    for exp_folder in sorted(experiments_dir.iterdir()):
        if exp_folder.is_dir():
            report_path = exp_folder / 'medical_metrics_report.txt'
            config_path = exp_folder / 'config.txt'

            # Если нет отчета, тогда точно пропускаем
            if not report_path.exists():
                continue

            exp_data = parse_report(report_path)
            exp_data['Experiment_Folder'] = exp_folder.name
            
            # ЗАЩИТА: Если конфига нет, скрипт больше не падает и не пропускает папку
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8', errors='ignore') as f:
                    exp_data['Config'] = f.read().strip()
            else:
                exp_data['Config'] = "Config Missing"

            all_results.append(exp_data)
            
            # Простая проверка на битые данные
            if exp_data.get('MCC') == 'N/A':
                print(f"  [ВНИМАНИЕ] В папке {exp_folder.name} есть метрики N/A. Проверьте формат файла.")
            else:
                print(f"  [+] Успешно обработан: {exp_folder.name}")

    if not all_results:
        print("\n[!] Нет отчетов для сбора.")
        return

    output_file = 'all_experiments_summary.csv'
    
    base_fields = [
        'Experiment_Folder', 'Config', 
        'Balanced_Accuracy', 'MCC', 
        'F1_Macro', 'F1_Weighted',
        'Accuracy', 'Sensitivity_Global', 'Specificity_Global', 
        'PR_AUC', 'ROC_AUC_Macro', 'ROC_AUC_Weighted'
    ]
    
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    class_fields = []
    for cls in classes:
        class_fields.extend([f'{cls}_Recall', f'{cls}_Specificity', f'{cls}_Precision', f'{cls}_F1'])

    fieldnames = base_fields + class_fields

    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        for row in all_results:
            clean_row = {field: row.get(field, 'N/A') for field in fieldnames}
            writer.writerow(clean_row)

    print(f"\n[SUCCESS] Таблица успешно сохранена в {output_file}")

if __name__ == '__main__':
    main()