from pathlib import Path
import cv2
from tqdm import tqdm

def prepare_for_gan_raw(base_src_dir, base_out_dir, classes, size=(256, 256)):
    base_src = Path(base_src_dir)
    base_out = Path(base_out_dir)

    for cls in classes:
        src_path = base_src / cls
        out_path = base_out / f"{cls}_raw"

        if not src_path.exists():
            print(f"[!] Папка {src_path} не найдена")
            continue

        out_path.mkdir(parents=True, exist_ok=True)
        images = list(src_path.glob("*.jpg")) + list(src_path.glob("*.png")) + list(src_path.glob("*.jpeg"))

        print(f"\n[*] {cls}: найдено {len(images)} изображений")

        for img_path in tqdm(images, desc=f"prepare {cls}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            m = min(h, w)
            x0 = w // 2 - m // 2
            y0 = h // 2 - m // 2
            img = img[y0:y0+m, x0:x0+m]
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

            cv2.imwrite(str(out_path / f"{img_path.stem}.png"), img)

if __name__ == "__main__":
    classes = ["akiec", "bcc", "df", "vasc"]
    prepare_for_gan_raw(
        base_src_dir="dataset/train",
        base_out_dir="gan_data_raw",
        classes=classes,
        size=(256, 256),
    )