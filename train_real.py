#!/usr/bin/env python3
"""用手工整理的数据训练 MobileNetV2 三分类模型。"""

import random
from pathlib import Path
from urllib.error import URLError

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASSES = ["two_wheel", "three_wheel", "four_wheel", "other"]
CLASS_NAMES = ["二轮车", "三轮车", "四轮车", "其他"]
DATA_DIR = Path("data")
MODEL_PATH = Path("wheelnet_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-3
SEED = 42

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
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
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    except (URLError, OSError, RuntimeError):
        print("warning: failed to load pretrained weights, using random init")
        model = models.mobilenet_v2(weights=None)
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASSES))
    return model


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
                image = Image.open(path).convert("RGB")
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


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_ds = VehicleDataset(DATA_DIR / "train", train_transform)
    val_ds = VehicleDataset(DATA_DIR / "val", val_transform)

    if len(train_ds) == 0:
        raise SystemExit("data/train 里没有可用图片")
    if len(val_ds) == 0:
        raise SystemExit("data/val 里没有可用图片，请先运行 split_manual_data.py")

    counts = [0] * len(CLASSES)
    for _, label in train_ds.samples:
        counts[label] += 1
    missing = [CLASS_NAMES[i] for i, c in enumerate(counts) if c == 0]
    if missing:
        raise SystemExit(
            f"以下类别在 data/train 下没有图片：{missing}，"
            "请先把图片放到 data/raw/<class>/ 并重新运行 split_manual_data.py"
        )
    weights = torch.tensor([1.0 / max(c, 1) for c in counts], dtype=torch.float32, device=DEVICE)
    weights = weights / weights.sum() * len(CLASSES)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_skip_bad,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_skip_bad,
    )

    model = make_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)

    best_acc = 0.0
    best_cm = None
    for epoch in range(1, EPOCHS + 1):
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

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        train_loss = running_loss / seen if seen else 0.0
        val_acc, val_cm = evaluate(model, val_loader)
        per_class = []
        for i, name in enumerate(CLASS_NAMES):
            total_i = val_cm[i].sum().item()
            acc_i = val_cm[i, i].item() / total_i if total_i else 0.0
            per_class.append(f"{name}={acc_i:.2f}")
        print(
            f"epoch {epoch}/{EPOCHS} train_loss={train_loss:.4f} "
            f"val_acc={val_acc:.4f} [{' '.join(per_class)}]"
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

    print(f"best_val_acc={best_acc:.4f}")
    print(f"saved: {MODEL_PATH}")

    if best_cm is not None:
        print("\n混淆矩阵（行=真实, 列=预测）:")
        print("           " + "  ".join(f"{n}" for n in CLASS_NAMES))
        for i, name in enumerate(CLASS_NAMES):
            row = "  ".join(f"{best_cm[i, j].item():4d}" for j in range(len(CLASSES)))
            print(f"  {name}     {row}")


if __name__ == "__main__":
    main()
