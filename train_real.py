#!/usr/bin/env python3
"""用手工整理的数据训练 MobileNetV2 四分类模型 — 渐进解冻 + 强化增强版。"""

import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASSES = ["two_wheel", "three_wheel", "four_wheel", "other"]
CLASS_NAMES = ["二轮车", "三轮车", "四轮车", "其他"]
# 三轮车（index=1）是困难类别，手动抬升其损失权重
CLASS_LOSS_BOOST = [1.0, 1.5, 1.0, 1.0]

DATA_DIR = Path("data")
MODEL_PATH = Path("wheelnet_model.pth")
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
BATCH_SIZE = 16
NUM_WORKERS = 2
EPOCHS = 18
SEED = 42

# ── 增强 pipeline ──────────────────────────────────────────
# 训练：强几何增强，迫使模型关注轮子数量而非整体轮廓
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def is_image_file(path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_model():
    """创建模型并冻结 backbone（初始阶段只训练分类头）。"""
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    # 冻结整个 backbone
    for param in model.features.parameters():
        param.requires_grad = False
    # 替换分类头
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASSES))
    return model


def set_backbone_trainable(model, stage: int):
    """
    渐进解冻 backbone：
    stage 1 — 只训练分类头（初始状态）
    stage 2 — 解冻 features[14:]（深层 block，感受野大，决定高级语义）
    stage 3 — 解冻 features[7:]（中层 block，学习中级几何特征）
    """
    if stage == 1:
        for param in model.features.parameters():
            param.requires_grad = False
    elif stage == 2:
        for i, child in enumerate(model.features):
            for param in child.parameters():
                param.requires_grad = i >= 14
    elif stage == 3:
        for i, child in enumerate(model.features):
            for param in child.parameters():
                param.requires_grad = i >= 7


def _count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class VehicleDataset(Dataset):
    def __init__(self, root, transform):
        self.samples = []
        self.transform = transform
        for label, class_name in enumerate(CLASSES):
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and is_image_file(path):
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        n = len(self.samples)
        for offset in range(n):
            path, label = self.samples[(index + offset) % n]
            try:
                img = Image.open(path)
                if img.mode == "P" and "transparency" in img.info:
                    img = img.convert("RGBA")
                image = img.convert("RGB")
            except (OSError, UnidentifiedImageError):
                continue
            return self.transform(image), label
        raise RuntimeError("所有样本都无法读取")


def collate_skip_bad(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels)


def evaluate(model, loader):
    model.eval()
    n_classes = len(CLASSES)
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            images, labels = batch
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                cm[t, p] += 1
    total = cm.sum().item()
    correct = cm.diag().sum().item()
    acc = correct / total if total else 0.0
    return acc, cm


def _per_class_metrics(cm):
    """从混淆矩阵计算每类的 Precision / Recall / F1。"""
    n = cm.shape[0]
    metrics = []
    for i in range(n):
        tp = cm[i, i].item()
        total_true = cm[i].sum().item()
        total_pred = cm[:, i].sum().item()
        rec = tp / total_true if total_true else 0.0
        prec = tp / total_pred if total_pred else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        metrics.append((prec, rec, f1))
    return metrics


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_ds = VehicleDataset(DATA_DIR / "train", train_transform)
    val_ds = VehicleDataset(DATA_DIR / "val", val_transform)

    if len(train_ds) == 0:
        raise SystemExit("data/train 里没有可用图片")
    if len(val_ds) == 0:
        raise SystemExit("data/val 里没有可用图片，请先运行 split_manual_data.py")

    # 样本统计
    counts = [0] * len(CLASSES)
    for _, label in train_ds.samples:
        counts[label] += 1
    print("训练集样本分布:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {counts[i]} 张")

    missing = [CLASS_NAMES[i] for i, c in enumerate(counts) if c == 0]
    if missing:
        raise SystemExit(f"以下类别在 data/train 下没有图片：{missing}")

    # 损失权重：样本数反比 × 手动 boost（三轮车加重）
    boost = torch.tensor(CLASS_LOSS_BOOST, dtype=torch.float32)
    inv_counts = torch.tensor([1.0 / max(c, 1) for c in counts], dtype=torch.float32)
    loss_weights = inv_counts * boost
    loss_weights = loss_weights / loss_weights.sum() * len(CLASSES)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_skip_bad,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_skip_bad,
    )

    model = make_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(DEVICE))
    best_acc = 0.0
    best_cm = None

    # ── 三阶段渐进训练 ─────────────────────────────────
    # 阶段 1：epoch 1-6，只训练分类头，LR=1e-3
    # 阶段 2：epoch 7-12，解冻深层 backbone，LR=1e-4
    # 阶段 3：epoch 13-18，解冻中层 backbone，LR=1e-5
    stages = [
        (1, 6, 1, 1e-3),
        (7, 12, 2, 1e-4),
        (13, EPOCHS, 3, 1e-5),
    ]

    global_epoch = 0
    for stage_start, stage_end, stage, base_lr in stages:
        set_backbone_trainable(model, stage)
        trainable = _count_trainable(model)
        print(f"\n── 阶段 {stage} (epoch {stage_start}-{stage_end}) "
              f"LR={base_lr:.0e} 可训练参数={trainable:,} ──")

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=base_lr)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=(stage_end - stage_start + 1) * len(train_loader),
            eta_min=base_lr * 0.1,
        )

        for epoch in range(stage_start, stage_end + 1):
            global_epoch += 1
            model.train()
            running_loss = 0.0
            seen = 0

            for batch in train_loader:
                if batch is None:
                    continue
                images, labels = batch
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                running_loss += loss.item() * labels.size(0)
                seen += labels.size(0)

            train_loss = running_loss / seen if seen else 0.0
            val_acc, val_cm = evaluate(model, val_loader)
            per_class = _per_class_metrics(val_cm)

            # 打印每类指标
            parts = []
            for i, name in enumerate(CLASS_NAMES):
                prec, rec, f1 = per_class[i]
                parts.append(f"{name}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
            print(
                f"epoch {global_epoch:2d}/{EPOCHS} "
                f"train_loss={train_loss:.4f} val_acc={val_acc:.4f} "
                f"[{' | '.join(parts)}]"
            )

            if val_acc >= best_acc:
                best_acc = val_acc
                best_cm = val_cm
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "classes": CLASSES,
                        "display_classes": CLASS_NAMES,
                    },
                    MODEL_PATH,
                )

    # ── 最终报告 ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"best_val_acc={best_acc:.4f}")
    print(f"saved: {MODEL_PATH}")

    if best_cm is not None:
        per_class = _per_class_metrics(best_cm)
        total = best_cm.sum().item()
        print("\n混淆矩阵（行=真实, 列=预测）:")
        header = "           " + "  ".join(f"{n:^6}" for n in CLASS_NAMES)
        print(header)
        for i, name in enumerate(CLASS_NAMES):
            row = "  ".join(f"{best_cm[i, j].item():6d}" for j in range(len(CLASSES)))
            rec = best_cm[i, i].item() / best_cm[i].sum().item() if best_cm[i].sum() else 0
            print(f"  {name:　<6}  {row}  (recall={rec:.4f})")

        print("\n          " + "  ".join(f"{n:^6}" for n in CLASS_NAMES))
        precisions = []
        for j in range(len(CLASSES)):
            col_sum = best_cm[:, j].sum().item()
            prec = best_cm[j, j].item() / col_sum if col_sum else 0
            precisions.append(f"{prec:.4f}")
            print(f"  prec    {'':>6}" * j + f"  {prec:.4f}" + f"{'':>6}" * (len(CLASSES) - j - 1))

        macro_prec = sum(p["prec"] for p in [
            {"prec": best_cm[i,i].item()/(best_cm[:,i].sum().item() or 1)}
            for i in range(len(CLASSES))
        ]) / len(CLASSES)
        macro_rec = sum(
            best_cm[i,i].item() / (best_cm[i].sum().item() or 1)
            for i in range(len(CLASSES))
        ) / len(CLASSES)
        print(f"\n总计 {int(total)} 张, 正确 {int(best_cm.diag().sum().item())} 张")
        print(f"Accuracy={best_acc:.4f}  Macro-P={macro_prec:.4f}  Macro-R={macro_rec:.4f}")


if __name__ == "__main__":
    main()
