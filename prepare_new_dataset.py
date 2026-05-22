from __future__ import annotations

import os
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def remove_hair(img_bgr: np.ndarray) -> np.ndarray:
  """
  Удаление волос по схеме:
  grayscale -> black-hat -> blur -> Otsu threshold -> inpainting.
  Работает устойчивее, чем фиксированный порог.
  """
  gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

  # Элипс лучше подходит для тонких линий и кривых волос
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
  blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

  # Слегка сгладим шум перед порогом
  blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)

  # Адаптивнее, чем fixed threshold=15
  _, hair_mask = cv2.threshold(
    blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
  )

  # Почистим маску от мусора
  hair_mask = cv2.medianBlur(hair_mask, 3)
  hair_mask = cv2.dilate(hair_mask, np.ones((3, 3), np.uint8), iterations=1)

  # Inpainting
  return cv2.inpaint(img_bgr, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def apply_clahe(img_bgr: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
  """
  Мягкая нормализация освещения через CLAHE по L-каналу в LAB.
  Обычно безопаснее для дерматоскопии, чем Gray World.
  """
  lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
  l, a, b = cv2.split(lab)

  clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
  l2 = clahe.apply(l)

  lab2 = cv2.merge((l2, a, b))
  return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def resize_if_needed(img_bgr: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
  """
  Опциональное приведение к фиксированному размеру.
  size задаётся как (width, height).
  """
  if size is None:
    return img_bgr
  return cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)


def process_single_image(args) -> str:
  src_path, dst_path, resize_size = args

  try:
    if dst_path.exists():
      return "skipped"

    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
      return f"failed_read: {src_path}"

    # 1) Удаляем волосы
    img = remove_hair(img)

    # 2) Мягко выравниваем освещение / контраст
    img = apply_clahe(img)

    # 3) Опционально изменяем размер
    img = resize_if_needed(img, resize_size)

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Для JPEG можно чуть снизить потери качества
    if dst_path.suffix.lower() in {".jpg", ".jpeg"}:
      ok = cv2.imwrite(
        str(dst_path),
        img,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
      )
    else:
      ok = cv2.imwrite(str(dst_path), img)

    if not ok:
      return f"failed_write: {src_path}"

    return "ok"

  except Exception as e:
    return f"error: {src_path} -> {e}"


def build_tasks(src_dir: Path, dst_dir: Path, resize_size: tuple[int, int] | None):
  tasks = []

  for root, _, files in os.walk(src_dir):
    root_path = Path(root)
    rel_path = root_path.relative_to(src_dir)
    target_root = dst_dir / rel_path
    target_root.mkdir(parents=True, exist_ok=True)

    for file in files:
      if Path(file).suffix.lower() in IMG_EXTENSIONS:
        src_path = root_path / file
        dst_path = target_root / file
        tasks.append((src_path, dst_path, resize_size))

  return tasks


def main():
  src_dir = Path("dataset")
  dst_dir = Path("dataset_preprocessed_v2")

  # Если нужен фиксированный размер, раскомментируй:
  # resize_size = (224, 224)
  resize_size = None

  if not src_dir.exists():
    print("Исходная папка dataset/ не найдена!")
    return

  print("[*] Начинаем предобработку HAM10000...")
  print(f"[*] Исходная папка: {src_dir}")
  print(f"[*] Целевая папка: {dst_dir}")

  tasks = build_tasks(src_dir, dst_dir, resize_size)
  print(f"[*] Найдено {len(tasks)} изображений")

  # Для CPU-heavy обработки процессы обычно лучше потоков
  workers = max(1, os.cpu_count() or 1)

  ok_count = 0
  fail_count = 0

  with ProcessPoolExecutor(max_workers=workers) as executor:
    for result in tqdm(executor.map(process_single_image, tasks),
            total=len(tasks),
            desc="Обработка"):
      if result == "ok" or result == "skipped":
        ok_count += 1
      else:
        fail_count += 1
        logging.warning(result)

  print(f"\n[SUCCESS] Обработка завершена.")
  print(f"[*] Успешно/пропущено: {ok_count}")
  print(f"[*] Ошибок: {fail_count}")
  print(f"[*] Результат сохранён в: {dst_dir}")


if __name__ == "__main__":
    main()