#!/usr/bin/env python3
"""从 data/raw 稳定生成 data/train 和 data/val。"""

import random
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError


CLASSES = ["two_wheel", "three_wheel", "four_wheel", "other"]
RAW_DIR = Path("data/raw")
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
VAL_RATIO = 0.2
SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_valid_image(path):
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
    except (OSError, UnidentifiedImageError):
        return False
    return True


def reset_output_dirs():
    shutil.rmtree(TRAIN_DIR, ignore_errors=True)
    shutil.rmtree(VAL_DIR, ignore_errors=True)
    for class_name in CLASSES:
        (TRAIN_DIR / class_name).mkdir(parents=True, exist_ok=True)
        (VAL_DIR / class_name).mkdir(parents=True, exist_ok=True)


def split_class(class_name):
    raw_class_dir = RAW_DIR / class_name
    if not raw_class_dir.exists():
        print(f"{class_name}: data/raw 下没有这个目录")
        return

    train_class_dir = TRAIN_DIR / class_name
    val_class_dir = VAL_DIR / class_name

    images = [
        p for p in sorted(raw_class_dir.iterdir())
        if p.is_file() and is_valid_image(p)
    ]
    if not images:
        print(f"{class_name}: 没有可用图片")
        return

    random.shuffle(images)
    val_count = max(1, int(len(images) * VAL_RATIO))
    val_images = set(images[:val_count])

    for src in images:
        dst_dir = val_class_dir if src in val_images else train_class_dir
        shutil.copy2(src, dst_dir / src.name)

    print(f"{class_name}: train={len(images) - val_count}, val={val_count}")


def main():
    random.seed(SEED)
    if not RAW_DIR.exists():
        raise SystemExit("data/raw 不存在，请先把图片放到 data/raw/{two_wheel,three_wheel,four_wheel}/")

    reset_output_dirs()
    for class_name in CLASSES:
        split_class(class_name)


if __name__ == "__main__":
    main()
