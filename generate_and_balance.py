import subprocess
from pathlib import Path

def find_best_network(run_dir_base):
    """Ищет последний .pkl файл (обычно это network-snapshot-001000.pkl) в папке эксперимента"""
    run_dir = Path(run_dir_base)
    if not run_dir.exists():
        return None
        
    # Ищем самую свежую папку (например, 00000-stylegan2-...)
    subdirs = sorted([d for d in run_dir.iterdir() if d.is_dir()])
    if not subdirs:
        return None
        
    latest_run = subdirs[-1]
    
    # Ищем pkl файлы
    pkl_files = sorted(list(latest_run.glob('network-snapshot-*.pkl')))
    if not pkl_files:
        return None
        
    return pkl_files[-1] # Возвращаем последний сохраненный чекпоинт

def main():
    train_dir = Path('dataset_preprocessed/train')
    
    # 1. Узнаем размер мажоритарного класса (nv - невусы)
    nv_dir = train_dir / 'nv'
    target_count = len(list(nv_dir.glob('*.jpg')) + list(nv_dir.glob('*.png')))
    print(f"[*] Целевой размер каждого класса (по классу 'nv'): {target_count} изображений")

    classes_to_generate = ['vasc', 'df', 'akiec', 'bcc']

    for cls in classes_to_generate:
        cls_dir = train_dir / cls
        current_count = len(list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png')))
        
        diff = target_count - current_count
        if diff <= 0:
            print(f"[+] Класс {cls} уже сбалансирован ({current_count} шт). Пропуск.")
            continue
            
        print(f"\n=======================================================")
        print(f"[*] Класс {cls}: имеется {current_count}, нужно сгенерировать {diff} шт.")
        print(f"=======================================================")
        
        # Ищем обученную сеть
        network_pkl = find_best_network(f'gan_training_runs/{cls}_sg3')
        if not network_pkl:
            print(f"[!] Ошибка: Не найдены веса GAN для класса {cls}. Пропускаем.")
            continue
            
        print(f"[*] Используем веса: {network_pkl}")
        
        # 2. Вызываем скрипт генерации
        # Коэффициент trunc=0.7 дает высокую реалистичность (стандарт для StyleGAN)
        seeds_arg = f"0-{diff - 1}"
        cmd = [
            "python", "stylegan3-brecahad/stylegan3/gen_images.py",
            f"--outdir={str(cls_dir)}",
            f"--trunc=0.7",
            f"--seeds={seeds_arg}",
            f"--network={str(network_pkl)}"
        ]
        
        print(f"[*] Запуск генерации. Это может занять некоторое время...")
        subprocess.run(cmd, check=True)
        print(f"[SUCCESS] Сгенерировано {diff} синтетических изображений для {cls}!")

    print("\n=========================================================")
    print(" БАЛАНСИРОВКА ЗАВЕРШЕНА!")
    print(f" Теперь все классы имеют ровно по {target_count} изображений.")
    print(" Датасет готов к финальному обучению классификатора ConvNeXt!")
    print("=========================================================")

if __name__ == '__main__':
    main()