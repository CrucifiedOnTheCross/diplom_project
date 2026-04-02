import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
import random # Добавлен модуль для случайного выбора

# Импорт библиотеки Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Твоя архитектура
from model import SkinLesionClassifier

def load_image(image_path, device):
    """Загрузка и препроцессинг изображения"""
    img = Image.open(image_path).convert('RGB')
    
    # Сохраняем оригинал для визуализации (отмасштабированный до 256x256)
    img_resized = img.resize((256, 256))
    rgb_img = np.float32(img_resized) / 255.0
    
    # Препроцессинг для модели
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    return rgb_img, input_tensor

def find_target_layer(model):
    """Автоматический поиск правильного слоя для Grad-CAM в ConvNeXt"""
    try:
        # Для torchvision ConvNeXt
        return [model.backbone.features[-1][-1]]
    except AttributeError:
        pass
    
    try:
        # Для timm ConvNeXt
        return [model.backbone.stages[-1].blocks[-1]]
    except AttributeError:
        print("[!] Не удалось автоматически определить target_layer. Убедись, что используешь ConvNeXt.")
        return None

def get_random_test_images(valid_dir, num_per_class=2):
    """Случайно выбирает num_per_class картинок из каждого класса валидационной выборки"""
    test_images = []
    valid_path = Path(valid_dir)
    
    if not valid_path.exists():
        print(f"[!] Ошибка: Папка валидации не найдена по пути: {valid_dir}")
        return test_images
        
    for class_dir in valid_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            # Собираем все картинки в папке класса
            all_images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            if all_images:
                # Выбираем случайные N картинок (или меньше, если их не хватает)
                selected = random.sample(all_images, min(num_per_class, len(all_images)))
                for img_path in selected:
                    test_images.append((str(img_path), class_name))
                    
    return test_images

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Классы датасета
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    # Папка с экспериментами и папка для сохранения
    experiments_dir = Path("experiments")
    out_dir = Path("gradcam_results")
    out_dir.mkdir(exist_ok=True)
    
    # =====================================================================
    # СЛУЧАЙНЫЙ ВЫБОР КАРТИНОК ИЗ ВАЛИДАЦИИ
    # =====================================================================
    valid_directory = "dataset_preprocessed/valid"
    images_per_class = 2 # По 2 картинки из каждого из 7 классов = 14 картинок всего
    
    print(f"[*] Сбор случайных изображений из {valid_directory}...")
    test_images = get_random_test_images(valid_directory, num_per_class=images_per_class)
    
    if not test_images:
        print("[!] Ошибка: Не удалось собрать тестовые изображения. Проверь путь к валидации.")
        return

    # Ищем все файлы best_model.pth во всех подпапках experiments/
    model_paths = list(experiments_dir.glob("*/best_model.pth"))
    
    if not model_paths:
        print(f"[!] В папке {experiments_dir} не найдено ни одного файла best_model.pth")
        return

    print(f"[*] Найдено экспериментов: {len(model_paths)}")
    print(f"[*] Выбрано случайных тестовых изображений: {len(test_images)}")
    
    # Основной цикл по всем экспериментам
    for model_path in tqdm(model_paths, desc="Анализ экспериментов"):
        exp_name = model_path.parent.name
        
        # Инициализация чистой модели
        model = SkinLesionClassifier(num_classes=len(classes)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        target_layers = find_target_layer(model)
        if not target_layers:
            continue
            
        cam = cam = GradCAM(model=model, target_layers=target_layers)
        
        # Анализ картинок для текущей модели
        for img_path, true_label_name in test_images:
            path_obj = Path(img_path)
                
            rgb_img, input_tensor = load_image(img_path, device)
            
            # Получаем предсказание
            with torch.no_grad():
                output = model(input_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                pred_label_name = classes[pred_idx]
                
                probs = torch.softmax(output, dim=1)[0]
                confidence = probs[pred_idx].item() * 100
                
            # Генерация тепловой карты для предсказанного класса
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            # Визуализация
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(rgb_img)
            axes[0].set_title(f"Оригинал\nTrue: {true_label_name}")
            axes[0].axis('off')
            
            axes[1].imshow(cam_image)
            title_color = 'green' if true_label_name == pred_label_name else 'red'
            axes[1].set_title(f"Grad-CAM ({exp_name})\nPred: {pred_label_name} ({confidence:.1f}%)", color=title_color)
            axes[1].axis('off')
            
            plt.tight_layout()
            
            # Сохранение (Имя файла содержит: имя эксперимента, правильный класс, предсказанный класс и имя картинки)
            short_exp_name = exp_name.split("_", 2)[-1] # Убираем дату/время из имени для краткости
            save_name = f"{short_exp_name}_{true_label_name}_pred_{pred_label_name}_{path_obj.stem}.png"
            plt.savefig(out_dir / save_name, dpi=150)
            plt.close()

    print("\n[SUCCESS] Визуализация завершена! Проверьте папку 'gradcam_results'.")

if __name__ == '__main__':
    main()