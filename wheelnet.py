#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wheelnet.py — 基于 MobileNetV2 的二/三/四轮车辆分类
竞赛演示版（内置 3 类测试图片生成器）

使用方法：
    python3 wheelnet.py                  # 运行演示
    python3 wheelnet.py /path/to/img.jpg  # 识别单张图片
    python3 wheelnet.py --image /path/to/img.jpg
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ── 配置 ──────────────────────────────────────────────
CLASS_CODES = ["two_wheel", "three_wheel", "four_wheel", "other"]
DISPLAY_CLASSES = ["二轮车", "三轮车", "四轮车", "其他"]
DISPLAY_TO_CODE = dict(zip(DISPLAY_CLASSES, CLASS_CODES))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_PATH = Path("wheelnet_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 模型定义 ──────────────────────────────────────────

def create_model(pretrained=False):
    """创建 MobileNetV2 + 分类头（输出维度跟随 DISPLAY_CLASSES）"""
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(DISPLAY_CLASSES))
    return model


# ── 图片预处理 ────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ── 工具函数 ──────────────────────────────────────────

def load_image(image_path):
    """加载单张图片并预处理"""
    with Image.open(image_path) as img:
        if img.mode == "P" and "transparency" in img.info:
            img = img.convert("RGBA")
        return transform(img.convert("RGB")).unsqueeze(0)  # 加 batch 维度


def predict(model, image_tensor):
    """推理：返回类别名和置信度"""
    model.eval()
    image_tensor = image_tensor.to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return DISPLAY_CLASSES[pred.item()], conf.item()


def is_image_file(path):
    """判断路径是否为支持的图片文件。"""
    return path.suffix.lower() in IMAGE_SUFFIXES


def iter_image_paths(input_path, recursive=False):
    """返回单个图片或目录中的图片列表。"""
    path = Path(input_path).expanduser()
    if path.is_file():
        if not is_image_file(path):
            raise SystemExit(f"不是支持的图片文件: {path}")
        return [path]

    if not path.is_dir():
        raise SystemExit(f"图片或目录不存在: {path}")

    if recursive:
        images = [p for p in path.rglob("*") if p.is_file() and is_image_file(p)]
    else:
        images = [p for p in path.iterdir() if p.is_file() and is_image_file(p)]

    images = sorted(images)
    if not images:
        raise SystemExit(f"目录中没有支持的图片文件: {path}")
    return images


def read_image_metadata(image_path):
    """读取图片尺寸信息。"""
    with Image.open(image_path) as img:
        return {
            "width": img.width,
            "height": img.height,
            "depth": len(img.getbands()),
        }


def predict_image(model, image_path):
    """识别单张图片，返回可写入 JSON 的结构化结果。"""
    path = Path(image_path).expanduser()
    resolved = path.resolve()
    metadata = read_image_metadata(path)
    tensor = load_image(path)
    pred, conf = predict(model, tensor)

    return {
        "path": str(resolved),
        "filename": resolved.name,
        "folder": str(resolved.parent),
        **metadata,
        "class_code": DISPLAY_TO_CODE[pred],
        "class_name": pred,
        "confidence": round(conf, 6),
        "bbox": None,
    }


def build_json_report(input_path, recursive, results):
    """生成批量识别 JSON 报告。"""
    resolved_input = Path(input_path).expanduser().resolve()
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input": str(resolved_input),
        "recursive": recursive,
        "model_path": str(MODEL_PATH.resolve()) if MODEL_PATH.exists() else None,
        "device": str(DEVICE),
        "classes": [
            {"code": code, "name": name}
            for code, name in zip(CLASS_CODES, DISPLAY_CLASSES)
        ],
        "total": len(results),
        "results": results,
    }


def write_json_report(report, output_path):
    """把识别报告写入 JSON 文件。"""
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def load_model(require_trained=False):
    """加载训练模型；没有模型时可回退到预训练骨干。"""
    if MODEL_PATH.exists():
        print(f"[加载] 已有模型 {MODEL_PATH}")
        model = create_model(pretrained=False)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            saved = checkpoint.get("display_classes")
            if saved and list(saved) != DISPLAY_CLASSES:
                raise SystemExit(
                    f"模型类别不一致：checkpoint={list(saved)}，当前={DISPLAY_CLASSES}"
                )
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        return model.to(DEVICE).eval()

    if require_trained:
        raise SystemExit("未找到 wheelnet_model.pth，请先运行 train_real.py 生成模型")

    print("[训练] 演示模式：用默认预训练模型")
    print("[提示] 将 collect_data.py 下载的 data/ 目录放入后，运行 train_real.py 正式训练")

    model = create_model(pretrained=True)
    return model.to(DEVICE).eval()


def train_demo():
    """
    演示级训练：
    用内置的 9 张测试图片做 few-shot 微调，
    一旦你有了真实数据，替换掉 _demo_dataset 即可
    """
    return load_model(require_trained=False)


# ── 生成演示图片 ──────────────────────────────────────
# 用纯色块 + 形状模拟车辆轮廓（用于演示，不依赖下载）

def _make_demo_image(shape_type, size=(224, 224)):
    """生成简单的演示图片，模拟车辆形状"""
    from PIL import ImageDraw

    img = Image.new("RGB", size, (200, 200, 200))
    draw = ImageDraw.Draw(img)

    if shape_type == "two_wheel":
        # 模拟二轮车：两个圆圈 + 横杠
        draw.ellipse([60, 140, 90, 170], fill=(50, 50, 150), outline=(0, 0, 0))
        draw.ellipse([140, 140, 170, 170], fill=(50, 50, 150), outline=(0, 0, 0))
        draw.rectangle([85, 70, 140, 150], fill=(100, 100, 200))
        draw.rectangle([80, 55, 145, 75], fill=(80, 80, 180))
    elif shape_type == "three_wheel":
        # 模拟三轮车：两个后轮 + 一个前轮 + 货斗
        draw.ellipse([50, 150, 80, 180], fill=(150, 100, 50), outline=(0, 0, 0))
        draw.ellipse([140, 150, 170, 180], fill=(150, 100, 50), outline=(0, 0, 0))
        draw.ellipse([100, 170, 125, 195], fill=(150, 100, 50), outline=(0, 0, 0))
        draw.rectangle([70, 80, 150, 150], fill=(180, 130, 70))
        draw.rectangle([65, 65, 155, 85], fill=(160, 110, 50))
    else:
        # 模拟四轮车：四个轮子 + 车身
        draw.ellipse([40, 155, 70, 185], fill=(60, 60, 60), outline=(0, 0, 0))
        draw.ellipse([155, 155, 185, 185], fill=(60, 60, 60), outline=(0, 0, 0))
        draw.ellipse([40, 55, 70, 85], fill=(60, 60, 60), outline=(0, 0, 0))
        draw.ellipse([155, 55, 185, 85], fill=(60, 60, 60), outline=(0, 0, 0))
        draw.rectangle([55, 40, 170, 185], fill=(80, 130, 200))
        draw.rectangle([70, 30, 155, 50], fill=(70, 120, 190))

    return img


def run_demo(model):
    """运行演示"""
    print("\n" + "=" * 50)
    print("wheelnet 演示"
          "\n基于 MobileNetV2 的车辆轮数分类")
    print("=" * 50)
    print("\n说明：演示输入为合成色块，仅用于验证推理流程；预测结果不代表真实准确率。")
    print("      要看真实识别，请用 python3 wheelnet.py /path/to/image.jpg")

    types = ["two_wheel", "three_wheel", "four_wheel"]
    names = ["二轮车", "三轮车", "四轮车"]

    print(f"\n{'合成图样':<10} {'预测结果':<10} {'置信度':<10}")
    print("-" * 35)

    for t, name in zip(types, names):
        img = _make_demo_image(t)
        tensor = transform(img).unsqueeze(0)
        pred, conf = predict(model, tensor)
        print(f"{name + '形状':<10} {pred:<10} {conf:.2%}")

    print(f"\n设备: {DEVICE}")
    print(f"模型: MobileNetV2 (参数量: ~3.5M)")


def print_single_result(result):
    """按原来的格式输出单张识别结果。"""
    print("\n" + "=" * 50)
    print("wheelnet 单图识别")
    print("=" * 50)
    print(f"图片: {result['path']}")
    print(f"识别结果: {result['class_name']}")
    print(f"置信度: {result['confidence']:.2%}")
    print(f"尺寸: {result['width']}x{result['height']}x{result['depth']}")
    print(f"设备: {DEVICE}")


def print_batch_start(total):
    """输出批量识别开始信息。"""
    print("\n" + "=" * 50)
    print("wheelnet 批量识别")
    print("=" * 50)
    print(f"图片数量: {total}")
    print(f"设备: {DEVICE}")
    print("-" * 50)


def print_progress_current(index, total, image_path):
    """输出当前正在识别的图片。"""
    path = Path(image_path).expanduser().resolve()
    print(f"[{index}/{total}] 正在识别: {path}", flush=True)


def print_progress_result(index, total, result):
    """输出当前图片识别结果和完成进度。"""
    percent = index / total * 100
    print(
        f"[{index}/{total} {percent:6.2f}%] "
        f"识别结果: {result['class_name']} "
        f"置信度: {result['confidence']:.2%}",
        flush=True,
    )


def print_batch_done(results):
    """输出批量识别结束信息。"""
    print("-" * 50)
    print(f"识别完成: {len(results)} 张")


def run_image_cli(input_path, json_output=None, recursive=False):
    """识别单张图片或目录；可选写入 JSON 报告。"""
    image_paths = iter_image_paths(input_path, recursive=recursive)
    model = load_model(require_trained=True)
    batch_output = (
        json_output is not None
        or len(image_paths) > 1
        or Path(input_path).expanduser().is_dir()
    )

    if batch_output:
        print_batch_start(len(image_paths))

    results = []
    total = len(image_paths)
    for index, path in enumerate(image_paths, start=1):
        if batch_output:
            print_progress_current(index, total, path)
        result = predict_image(model, path)
        results.append(result)
        if batch_output:
            print_progress_result(index, total, result)

    if batch_output:
        print_batch_done(results)
    else:
        print_single_result(results[0])

    if json_output:
        report = build_json_report(input_path, recursive, results)
        output_path = write_json_report(report, json_output)
        print(f"\nJSON结果: {output_path}")


# ── 主入口 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="wheelnet 车辆轮数分类")
    parser.add_argument("image", nargs="?", help="要识别的图片或目录路径")
    parser.add_argument(
        "-i",
        "--image",
        "--image-path",
        "--input",
        dest="image_path",
        help="要识别的图片或目录路径",
    )
    parser.add_argument(
        "-o",
        "--json-output",
        help="把识别结果写入指定 JSON 文件",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="输入为目录时递归扫描子目录",
    )
    args = parser.parse_args()

    image_path = args.image_path or args.image
    if image_path:
        run_image_cli(image_path, json_output=args.json_output, recursive=args.recursive)
        return

    model = train_demo()
    run_demo(model)

    print("\n" + "-" * 50)
    print("下一步：")
    print("  1. 把图片放到 data/raw/{two_wheel,three_wheel,four_wheel}/")
    print("  2. 运行 split_manual_data.py 生成 train/val")
    print("  3. 运行 train_real.py 正式训练")
    print("  4. 运行 wheelnet.py / wheelnet.py -i 图片路径 识别单图")
    print("-" * 50)


if __name__ == "__main__":
    main()
