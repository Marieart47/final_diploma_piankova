"""
Утилита подготовки HR-изображений для задачи Super Resolution.

Поддерживаемые источники:
  - CelebA  : скачивается через torchvision, сохраняются центрированные кропы 178×178
  - DIV2K   : скачивается вручную с https://data.div2k.net/ , затем укажите --src_dir
  - Flickr2K: скачивается вручную, затем укажите --src_dir
  - custom   : любая папка с изображениями через --src_dir

Использование:

  # CelebA (скачивается автоматически, ~1.3 GB):
  python datasets/prepare_sr_data.py --source celeba --out_dir data/sr_hr/celeba --max_images 5000

  # DIV2K (HR-изображения скачайте вручную, укажите папку):
  python datasets/prepare_sr_data.py --source custom --src_dir /path/to/DIV2K_train_HR --out_dir data/sr_hr/div2k

  # Произвольная папка:
  python datasets/prepare_sr_data.py --source custom --src_dir /path/to/images --out_dir data/sr_hr/custom
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _save_celeba(out_dir: Path, max_images: int, data_root: str):
    from torchvision.datasets import CelebA
    print("Загрузка CelebA через torchvision (может занять несколько минут)...")
    ds = CelebA(root=data_root, split="train", target_type="attr", download=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(ds), max_images)
    for i in tqdm(range(n), desc="CelebA"):
        img, _ = ds[i]
        # CelebA возвращает PIL Image; центральный кроп 178×178 (уже в датасете)
        img.save(out_dir / f"{i:05d}.png")
    print(f"Сохранено {n} изображений в {out_dir}")


def _copy_custom(src_dir: Path, out_dir: Path, max_images: int, min_size: int):
    paths = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in _EXTS)
    if not paths:
        raise FileNotFoundError(f"Изображения не найдены в {src_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for p in tqdm(paths, desc="Копирование"):
        if saved >= max_images:
            break
        try:
            img = Image.open(p)
            if min(img.size) < min_size:
                continue
            dest = out_dir / f"{saved:05d}{p.suffix.lower()}"
            shutil.copy2(p, dest)
            saved += 1
        except Exception:
            continue
    print(f"Скопировано {saved} изображений в {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Подготовка HR-изображений для SR")
    parser.add_argument("--source", default="celeba",
                        choices=["celeba", "custom"],
                        help="Источник данных")
    parser.add_argument("--src_dir", default=None,
                        help="Папка с исходными HR-изображениями (для source=custom)")
    parser.add_argument("--out_dir", default="data/sr_hr",
                        help="Куда сохранять HR-изображения")
    parser.add_argument("--max_images", type=int, default=10000,
                        help="Максимальное количество изображений")
    parser.add_argument("--min_size", type=int, default=128,
                        help="Минимальная сторона изображения (px); меньше — пропускаем")
    parser.add_argument("--data_root", default="./data",
                        help="Корневая папка для torchvision (только для --source celeba)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.source == "celeba":
        _save_celeba(out_dir, args.max_images, args.data_root)
    elif args.source == "custom":
        if not args.src_dir:
            raise ValueError("Укажите --src_dir для source=custom")
        _copy_custom(Path(args.src_dir), out_dir, args.max_images, args.min_size)

    print("Готово.")


if __name__ == "__main__":
    main()
