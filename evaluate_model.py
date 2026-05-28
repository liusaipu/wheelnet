#!/usr/bin/env python3
"""加载 wheelnet_model.pth，在验证集上输出完整评估指标。"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_real import (
    CLASSES, CLASS_NAMES, DEVICE, DATA_DIR, BATCH_SIZE, SEED,
    VehicleDataset, val_transform, collate_skip_bad,
    make_model, evaluate, _per_class_metrics,
)


MODEL_PATH = Path("wheelnet_model.pth")


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"模型文件不存在: {MODEL_PATH}")

    val_ds = VehicleDataset(DATA_DIR / "val", val_transform)
    if len(val_ds) == 0:
        raise SystemExit("data/val 里没有可用图片，请先运行 split_manual_data.py")

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_skip_bad,
    )

    # 加载模型
    model = make_model().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    print(f"模型: {MODEL_PATH.resolve()}")
    print(f"验证集: {DATA_DIR.resolve() / 'val'}")

    val_acc, val_cm = evaluate(model, val_loader)
    per_class = _per_class_metrics(val_cm)
    total = val_cm.sum().item()
    correct = val_cm.diag().sum().item()

    # ── 整体指标 ───────────────────────────────────
    macro_prec = sum(p[0] for p in per_class) / len(per_class)
    macro_rec = sum(p[1] for p in per_class) / len(per_class)
    macro_f1 = sum(p[2] for p in per_class) / len(per_class)

    print(f"\n{'='*60}")
    print(f"  Accuracy        {val_acc:.4f}  ({int(correct)}/{int(total)})")
    print(f"  Macro-Precision {macro_prec:.4f}")
    print(f"  Macro-Recall    {macro_rec:.4f}")
    print(f"  Macro-F1        {macro_f1:.4f}")

    # ── 各类别指标 ─────────────────────────────────
    print(f"\n{'类别':　<8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'样本数':>8}")
    print("-" * 50)
    for i, name in enumerate(CLASS_NAMES):
        prec, rec, f1 = per_class[i]
        n = val_cm[i].sum().item()
        print(f"{name:　<8} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {n:>8d}")

    # ── 混淆矩阵 ───────────────────────────────────
    print(f"\n混淆矩阵（行=真实, 列=预测）:")
    header = "           " + "  ".join(f"{n:^6}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{val_cm[i, j].item():6d}" for j in range(len(CLASSES)))
        rec = val_cm[i, i].item() / val_cm[i].sum().item() if val_cm[i].sum() else 0
        print(f"  {name:　<6}  {row}  (recall={rec:.4f})")

    # 列方向 precision
    print(f"\n          {'':>6}" + "  ".join(f"{n:^6}" for n in CLASS_NAMES))
    prec_line = "  ".join(
        f"{val_cm[j, j].item() / val_cm[:, j].sum().item():.4f}"
        if val_cm[:, j].sum().item() else "  N/A "
        for j in range(len(CLASSES))
    )
    print(f"  prec    {'':>6}{prec_line}")


if __name__ == "__main__":
    main()
