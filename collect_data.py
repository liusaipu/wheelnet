#!/usr/bin/env python3
"""
车辆图片收集脚本

默认路线：
1. 从 COCO val2017 标注中下载二轮/四轮图片。
2. 三轮车使用多个公开图片搜索源兜底。
3. 下载后用 Pillow 校验并统一保存为 JPEG，再划分 train/val。

使用方法：
    python3 collect_data.py

输出：
    data/train/{two_wheel,three_wheel,four_wheel}/
    data/val/{two_wheel,three_wheel,four_wheel}/
"""

import json
import random
import shutil
import subprocess
import urllib.parse
import zipfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError


OUTPUT = Path("data")
TRAIN_RATIO = 0.85
USER_AGENT = "Mozilla/5.0 wheelnet-data-collector/1.0"
MIN_IMAGE_BYTES = 2_000
MIN_IMAGE_SIDE = 96
SEARCH_TIMEOUT = 12

CLASSES = ["two_wheel", "three_wheel", "four_wheel"]

# 每类目标数量。先保证能跑通，后续可以按时间增加。
TARGET = {
    "two_wheel": 150,
    "three_wheel": 100,
    "four_wheel": 150,
}

# COCO 官方标注包较大，下载一次即可复用。
ANN_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANN_JSON_URLS = [
    "https://huggingface.co/datasets/pcuenq/coco2017-instances/resolve/main/instances_val2017.json",
    "https://huggingface.co/datasets/PaDT-MLLM/COCO/resolve/main/instances_val2017.json",
    "https://huggingface.co/datasets/k-nick/coco2017/resolve/208965b41dd028343a537deaa3ee2b82110d2bf1/annotations/instances_val2017.json",
]

COCO_CATEGORY_IDS = {
    "two_wheel": [2, 4],      # bicycle, motorcycle
    "four_wheel": [3, 6, 8],  # car, bus, truck
}

SEARCH_KEYWORDS = {
    "two_wheel": [
        "bicycle vehicle",
        "motorcycle vehicle",
        "electric scooter vehicle",
    ],
    "three_wheel": [
        "tricycle vehicle",
        "three wheeled vehicle",
        "auto rickshaw",
        "tuk tuk",
        "cycle rickshaw",
        "三轮车",
        "三轮摩托",
        "三蹦子",
    ],
    "four_wheel": [
        "car vehicle",
        "truck vehicle",
        "bus vehicle",
        "van vehicle",
    ],
}


def run_curl(args, timeout):
    """运行 curl，返回 CompletedProcess。"""
    cmd = [
        "curl",
        "-s",
        "-S",
        "-k",
        "-L",
        "--retry",
        "2",
        "--connect-timeout",
        "10",
        "--max-time",
        str(timeout),
        "-A",
        USER_AGENT,
    ] + args
    return subprocess.run(cmd, capture_output=True, timeout=timeout + 15)


def fetch_text(url, timeout=20):
    """用 curl 拉取文本，失败时返回空字符串。"""
    try:
        result = run_curl([url], timeout=timeout)
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""

    if result.returncode != 0:
        return ""
    return result.stdout.decode("utf-8", errors="ignore")


def download_file(url, save_path, min_bytes=MIN_IMAGE_BYTES, timeout=45):
    """下载普通文件，使用临时文件避免留下坏文件。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_name(f"{save_path.name}.download")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        result = run_curl(["-o", str(tmp_path), url], timeout=timeout)
    except (subprocess.SubprocessError, FileNotFoundError):
        tmp_path.unlink(missing_ok=True)
        return False

    if result.returncode != 0 or not tmp_path.exists():
        tmp_path.unlink(missing_ok=True)
        return False

    if tmp_path.stat().st_size < min_bytes:
        tmp_path.unlink(missing_ok=True)
        return False

    tmp_path.replace(save_path)
    return True


def download_large_file(url, save_path, min_bytes, timeout=900):
    """下载大文件，支持断点续传。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_curl(["-C", "-", "-o", str(save_path), url], timeout=timeout)
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

    return (
        result.returncode == 0
        and save_path.exists()
        and save_path.stat().st_size >= min_bytes
    )


def normalize_image(path):
    """确认是可读图片，并统一转成 RGB JPEG。"""
    try:
        with Image.open(path) as img:
            img.load()
            if img.width < MIN_IMAGE_SIDE or img.height < MIN_IMAGE_SIDE:
                return False
            img.convert("RGB").save(path, "JPEG", quality=90)
    except (OSError, UnidentifiedImageError):
        return False
    return True


def download_image(url, save_path, timeout=45):
    if save_path.exists() and normalize_image(save_path):
        return True

    if not download_file(url, save_path, timeout=timeout):
        save_path.unlink(missing_ok=True)
        return False

    if not normalize_image(save_path):
        save_path.unlink(missing_ok=True)
        return False

    return True


def find_coco_json():
    candidates = [
        Path("coco_annotations") / "instances_val2017.json",
        Path("coco_annotations") / "annotations" / "instances_val2017.json",
        Path("annotations") / "instances_val2017.json",
    ]
    for path in candidates:
        if is_coco_annotation_file(path):
            return path
    return None


def is_coco_annotation_file(path):
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False

    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    return isinstance(data.get("images"), list) and isinstance(data.get("annotations"), list)


def extract_coco_json(zip_path, ann_dir):
    """从已有 COCO zip 中只提取 instances_val2017.json。"""
    if not zip_path.exists():
        return False
    if zip_path.stat().st_size < 100_000_000:
        return False

    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = [
                name for name in zf.namelist()
                if name.endswith("instances_val2017.json")
            ]
            if not names:
                return False
            zf.extract(names[0], ann_dir)
    except (OSError, RuntimeError, zipfile.BadZipFile):
        return False

    return find_coco_json() is not None


def ensure_coco_annotations():
    ann_json = find_coco_json()
    if ann_json:
        return ann_json

    ann_dir = Path("coco_annotations")
    ann_dir.mkdir(exist_ok=True)

    for json_url in ANN_JSON_URLS:
        ann_json = ann_dir / "instances_val2017.json"
        if is_coco_annotation_file(ann_json):
            return ann_json
        print(f"[COCO] 尝试下载精简标注: {json_url}")
        if download_file(
            json_url,
            ann_json,
            min_bytes=1_000_000,
            timeout=60,
        ):
            if is_coco_annotation_file(ann_json):
                return ann_json
            ann_json.unlink(missing_ok=True)

    for zip_path in [Path("annotations.zip"), ann_dir / "annotations_trainval2017.zip"]:
        if zip_path.exists():
            print(f"[COCO] 尝试使用已有标注包: {zip_path}")
            if extract_coco_json(zip_path, ann_dir):
                return find_coco_json()
            print(f"[COCO] {zip_path} 无法提取 instances_val2017.json，跳过")

    ann_zip = ann_dir / "annotations_trainval2017.zip"
    print("[COCO] 下载官方标注包（约 241MB，仅首次，支持断点续传）...")
    ok = download_large_file(
        ANN_URL,
        ann_zip,
        min_bytes=100_000_000,
        timeout=900,
    )
    if not ok:
        print("[COCO] 标注包下载失败，跳过 COCO 数据源")
        return None

    if not extract_coco_json(ann_zip, ann_dir):
        print("[COCO] 标注包损坏或不含 instances_val2017.json")
        return None

    return find_coco_json()


def download_coco_images_via_ids():
    """不依赖 pycocotools，直接解析 COCO val2017 标注并下载图片。"""
    ann_json = ensure_coco_annotations()
    if not ann_json:
        return {class_name: [] for class_name in COCO_CATEGORY_IDS}

    print("[COCO] 解析标注文件...")
    with ann_json.open(encoding="utf-8") as f:
        coco = json.load(f)

    img_map = {img["id"]: img for img in coco["images"]}
    downloaded = {}

    for class_name, ids in COCO_CATEGORY_IDS.items():
        target = TARGET[class_name]
        print(f"\n▶ {class_name} / COCO (目标 {target})")

        img_ids = {
            ann["image_id"]
            for ann in coco["annotations"]
            if ann["category_id"] in ids
        }
        img_ids = list(img_ids)
        random.shuffle(img_ids)

        save_dir = OUTPUT / "temp" / class_name
        save_dir.mkdir(parents=True, exist_ok=True)

        success = 0
        downloaded_files = []
        for img_id in img_ids:
            if success >= target:
                break

            info = img_map[img_id]
            fname = info["file_name"]
            save_path = save_dir / f"coco_{fname}"
            url = f"https://images.cocodataset.org/val2017/{fname}"

            if download_image(url, save_path, timeout=60):
                success += 1
                downloaded_files.append(save_path)
                if success % 30 == 0:
                    print(f"  已下载 {success}/{target}")

        print(f"  完成: {success} 张")
        downloaded[class_name] = downloaded_files

    return downloaded


def openverse_urls(keyword, pages=2, page_size=30):
    urls = []
    for page in range(1, pages + 1):
        query = urllib.parse.urlencode({
            "q": keyword,
            "page": page,
            "page_size": page_size,
        })
        text = fetch_text(
            f"https://api.openverse.org/v1/images/?{query}",
            timeout=SEARCH_TIMEOUT,
        )
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        for item in data.get("results", []):
            url = item.get("url") or item.get("thumbnail")
            if url:
                urls.append(url)
    return urls


def wikimedia_urls(keyword, limit=50):
    query = urllib.parse.urlencode({
        "action": "query",
        "generator": "search",
        "gsrsearch": keyword,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "format": "json",
    })
    text = fetch_text(
        f"https://commons.wikimedia.org/w/api.php?{query}",
        timeout=SEARCH_TIMEOUT,
    )
    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    urls = []
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        info = page.get("imageinfo", [{}])[0]
        mime = info.get("mime", "")
        url = info.get("url")
        if url and mime.startswith("image/") and "svg" not in mime:
            urls.append(url)
    return urls


def baidu_urls(keyword, pages=2, page_size=30):
    urls = []
    for page in range(pages):
        query = urllib.parse.urlencode({
            "tn": "resultjson_com",
            "word": keyword,
            "pn": page * page_size,
            "rn": page_size,
        })
        text = fetch_text(
            f"https://image.baidu.com/search/acjson?{query}",
            timeout=SEARCH_TIMEOUT,
        )
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        for item in data.get("data", []):
            url = item.get("thumbURL") or item.get("middleURL") or item.get("objURL")
            if url:
                urls.append(url)
    return urls


def collect_search_urls(class_name):
    seen = set()
    for keyword in SEARCH_KEYWORDS[class_name]:
        for fetcher in [openverse_urls, wikimedia_urls, baidu_urls]:
            for url in fetcher(keyword):
                if url.startswith("http") and url not in seen:
                    seen.add(url)
                    yield url


def download_from_search_sources(class_name, needed):
    """使用公开图片搜索源补齐某一类图片。"""
    print(f"\n▶ {class_name} / 搜图兜底 (还需 {needed})")

    save_dir = OUTPUT / "temp" / class_name
    save_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    start_idx = len(list(save_dir.glob("*.jpg")))
    urls = collect_search_urls(class_name)
    downloaded_files = []

    for i, url in enumerate(urls, start=start_idx):
        if success >= needed:
            break

        save_path = save_dir / f"{class_name}_web_{i:04d}.jpg"
        if download_image(url, save_path, timeout=45):
            success += 1
            downloaded_files.append(save_path)
            if success % 20 == 0:
                print(f"  已下载 {success}/{needed}")

    print(f"  完成: {success} 张")
    return downloaded_files


def split_data(class_name, images):
    train_dir = OUTPUT / "train" / class_name
    val_dir = OUTPUT / "val" / class_name
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_RATIO)
    if len(images) > 1:
        split_idx = min(max(1, split_idx), len(images) - 1)
    else:
        split_idx = len(images)

    for i, img in enumerate(images):
        dst = train_dir if i < split_idx else val_dir
        shutil.copy2(img, dst / img.name)

    print(f"  {class_name}: train={split_idx}, val={len(images) - split_idx}")


def main():
    random.seed(42)
    shutil.rmtree(OUTPUT / "temp", ignore_errors=True)

    print("=" * 50)
    print("车辆图片采集")
    print("=" * 50)

    downloaded = {class_name: [] for class_name in CLASSES}

    coco_downloaded = download_coco_images_via_ids()
    for class_name, images in coco_downloaded.items():
        downloaded[class_name] = images

    for class_name in CLASSES:
        current = len(downloaded[class_name])
        needed = max(0, TARGET[class_name] - current)
        if needed:
            downloaded[class_name].extend(download_from_search_sources(class_name, needed))

    print("\n" + "=" * 50)
    print("划分 train/val")
    print("=" * 50)
    for class_name in CLASSES:
        images = downloaded.get(class_name, [])
        if images:
            split_data(class_name, images)
        else:
            print(f"  {class_name}: 无图片，跳过")

    shutil.rmtree(OUTPUT / "temp", ignore_errors=True)

    print("\n" + "=" * 50)
    print("最终统计")
    print("=" * 50)
    total = 0
    for split in ["train", "val"]:
        for class_name in CLASSES:
            data_dir = OUTPUT / split / class_name
            count = len(list(data_dir.glob("*.*"))) if data_dir.exists() else 0
            total += count
            print(f"  {split}/{class_name}: {count} 张")

    print(f"\n总计: {total} 张")
    print("提示：自动采集后还需要人工删除错图、糊图和多车混杂图。")
    print("完成！")


if __name__ == "__main__":
    main()
