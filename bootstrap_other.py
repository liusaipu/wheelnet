#!/usr/bin/env python3
"""从 COCO val2017 抽取不含 wheelnet 三类车辆的图片，填充 data/raw/other/。

排除标注里含以下 COCO 类别的图片（与 wheelnet 三类直接重叠）：
    bicycle(2), car(3), motorcycle(4), bus(6), truck(8)

火车/飞机/船只等其他交通工具会被当作"其他"保留，这符合 wheelnet 只识别
路面轮式车辆的范围。
"""

import json
import random
import subprocess
from pathlib import Path

from PIL import Image, UnidentifiedImageError


TARGET = 200
SEED = 42
OUTPUT_DIR = Path("data/raw/other")
ANN_DIR = Path("coco_annotations")
ANN_JSON_NAME = "instances_val2017.json"
ANN_JSON_URLS = [
    "https://huggingface.co/datasets/pcuenq/coco2017-instances/resolve/main/instances_val2017.json",
    "https://huggingface.co/datasets/PaDT-MLLM/COCO/resolve/main/instances_val2017.json",
    "https://huggingface.co/datasets/k-nick/coco2017/resolve/208965b41dd028343a537deaa3ee2b82110d2bf1/annotations/instances_val2017.json",
]
VEHICLE_CATEGORIES = {2, 3, 4, 6, 8}
USER_AGENT = "Mozilla/5.0 wheelnet-other-bootstrap/1.0"
MIN_IMAGE_BYTES = 2_000
MIN_IMAGE_SIDE = 96


def run_curl(args, timeout):
    cmd = [
        "curl", "-s", "-S", "-k", "-L",
        "--retry", "2",
        "--connect-timeout", "10",
        "--max-time", str(timeout),
        "-A", USER_AGENT,
    ] + args
    return subprocess.run(cmd, capture_output=True, timeout=timeout + 15)


def download_file(url, save_path, min_bytes, timeout):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = save_path.with_name(f"{save_path.name}.download")
    if tmp.exists():
        tmp.unlink()
    try:
        result = run_curl(["-o", str(tmp), url], timeout=timeout)
    except (subprocess.SubprocessError, FileNotFoundError):
        tmp.unlink(missing_ok=True)
        return False
    if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size < min_bytes:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(save_path)
    return True


def normalize_image(path):
    try:
        with Image.open(path) as img:
            img.load()
            if img.width < MIN_IMAGE_SIDE or img.height < MIN_IMAGE_SIDE:
                return False
            img.convert("RGB").save(path, "JPEG", quality=90)
    except (OSError, UnidentifiedImageError):
        return False
    return True


def find_coco_json():
    candidates = [
        ANN_DIR / ANN_JSON_NAME,
        ANN_DIR / "annotations" / ANN_JSON_NAME,
        Path("annotations") / ANN_JSON_NAME,
    ]
    for p in candidates:
        if not p.exists() or p.stat().st_size < 1_000_000:
            continue
        try:
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data.get("images"), list) and isinstance(data.get("annotations"), list):
            return p
    return None


def ensure_coco_json():
    existing = find_coco_json()
    if existing:
        return existing

    ANN_DIR.mkdir(exist_ok=True)
    target = ANN_DIR / ANN_JSON_NAME
    for url in ANN_JSON_URLS:
        print(f"[COCO] 下载标注: {url}")
        if download_file(url, target, min_bytes=1_000_000, timeout=120):
            if find_coco_json():
                return target
            target.unlink(missing_ok=True)
    return None


def pick_non_vehicle_images(coco):
    vehicle_image_ids = {
        ann["image_id"]
        for ann in coco["annotations"]
        if ann["category_id"] in VEHICLE_CATEGORIES
    }
    annotated_ids = {ann["image_id"] for ann in coco["annotations"]}
    return [
        img for img in coco["images"]
        if img["id"] in annotated_ids and img["id"] not in vehicle_image_ids
    ]


def main():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ann_path = ensure_coco_json()
    if not ann_path:
        raise SystemExit("无法获取 COCO 标注文件，请检查网络")

    print(f"[COCO] 解析 {ann_path}")
    with ann_path.open(encoding="utf-8") as f:
        coco = json.load(f)

    candidates = pick_non_vehicle_images(coco)
    random.shuffle(candidates)
    print(f"[COCO] 候选非车图片：{len(candidates)} 张，目标下载 {TARGET}")

    success = 0
    for info in candidates:
        if success >= TARGET:
            break
        fname = info["file_name"]
        save_path = OUTPUT_DIR / f"coco_{fname}"

        if save_path.exists() and normalize_image(save_path):
            success += 1
            continue

        url = f"https://images.cocodataset.org/val2017/{fname}"
        if not download_file(url, save_path, min_bytes=MIN_IMAGE_BYTES, timeout=45):
            save_path.unlink(missing_ok=True)
            continue
        if not normalize_image(save_path):
            save_path.unlink(missing_ok=True)
            continue

        success += 1
        if success % 25 == 0:
            print(f"  已下载 {success}/{TARGET}")

    print(f"\n完成：{OUTPUT_DIR} 下共 {success} 张图片")
    print("提示：")
    print("  - COCO 标注是按显著物体打的，背景里有小车的图片可能漏过来 → 建议浏览一遍")
    print("  - COCO 里没有「玩具车 / 车的零件特写 / 婴儿车」等容易和真车混淆的样本，")
    print("    建议人工补充几十张这类图到 data/raw/other/")


if __name__ == "__main__":
    main()
