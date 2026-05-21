# 基于 MobileNet 的二轮、三轮、四轮车辆图像分类识别 — 项目方案

> 时间：2026-05-21

---

## 一、项目概述

**目标**：对监控场景下的车辆图像，自动识别为三类：

| 类别 | 包含 |
|------|------|
| 🏍️ 二轮 | 自行车、电动车、摩托车 |
| 🛺 三轮 | 三轮车、三轮摩托、三轮货运车 |
| 🚗 四轮 | 轿车、SUV、货车、客车等 |

**核心思路**：利用 MobileNet 自动学习车身轮廓、长宽比等深度特征，实现端到端分类，**不依赖手工设计的特征提取规则**。

---

## 二、技术方案

### 2.1 模型选型

以 MobileNet 为主干网络，替换最后的全连接层，输出 3 类。

**推荐方案：MobileNetV2**

| 模型 | 参数量 | 推理速度 | 适合场景 |
|------|--------|----------|----------|
| **MobileNetV2** | ~3.5M | ✅ 快 | 精度/速度最佳平衡，推荐 |
| MobileNetV3-Small | ~2.5M | 🚀 最快 | 低功耗/边缘设备 |
| ResNet50 | ~25M | 🐢 较慢 | 精度更高但非必需 |

**分类头设计（最后一个全连接层替换）：**

```
MobileNetV2 backbone
  → 7×7×1280 特征图
  → Global Average Pooling
  → FC(512) + ReLU + Dropout(0.2)
  → FC(3) + Softmax
```

### 2.2 核心思想说明

本项目名称为 **"基于 MobileNet 深度特征的车辆轮式分类"**，点明：
- ✅ **MobileNet 自动学习**：CNN 自动从像素中获取车轮数量、车身比例、骨架形态等判别性特征
- ❌ **非传统方法**：不采用二值化→找轮廓→算长宽比→规则判断的做法（受光照/遮挡影响大，鲁棒性差）

### 2.3 数据准备

**这是项目的核心难点**——公开数据集极少直接标注"轮数"。

#### 数据来源

| 来源 | 说明 | 预计可用量 |
|------|------|------------|
| BIT-Vehicle | 监控视角，按车型映射轮数 | ~9,000 张 |
| VehicleID / CompCars | 自然场景，按车型映射 | ~20,000+ |
| COCO 筛选 | 利用 bicycle/motorcycle/car/truck 等标签 | 大量可用 |
| UA-DETRAC | 监控视角车辆检测数据集 | ~8,000+ |
| 自行爬取 | 百度/Google 搜图补充 | 补充用 |

#### 数据增强（监控场景专用）

```python
import albumentations as A

train_transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),   # 光照变化
    A.GaussNoise(p=0.3),                  # 监控传感器噪声
    A.MotionBlur(p=0.3),                  # 运动模糊
    A.RandomShadow(p=0.2),                # 阴影遮挡
    A.Rotate(limit=15, p=0.5),            # 小角度旋转
    A.Perspective(scale=0.05, p=0.3),     # 透视畸变
    A.Resize(224, 224),
    A.Normalize(),
])
```

### 2.4 训练策略

**阶段一：冻结 backbone（~10 epochs）**
- 加载 ImageNet 预训练的 MobileNetV2
- 冻结所有 backbone 层
- 只训练新加的 FC 分类头
- 优化器：AdamW，lr=1e-3

**阶段二：解冻微调（~20 epochs）**
- 解冻 backbone 最后几层
- 优化器：AdamW，lr=1e-5（更小学习率）
- 学习率调度：Cosine Annealing

**Loss 设计**
- 基础：CrossEntropyLoss
- 进阶：LabelSmoothing(0.1) → 防过拟合
- 不均衡场景：FocalLoss(γ=2.0)

**类别不平衡处理**
- 三轮样本可能远少于二轮和四轮
- 使用 WeightedRandomSampler 重采样
- 或 FocalLoss 自动降权易分类样本

**监控指标**
- Accuracy（整体 + 每类别）
- Confusion Matrix（重点看三轮是否被误分）
- Precision / Recall / F1-score

### 2.5 推理部署

```
输入（监控帧）
  → [可选] YOLOv8 等检测器提取车辆框
  → 裁剪车辆区域
  → MobileNetV2 分类器
  → 输出 {二轮: 0.92, 三轮: 0.05, 四轮: 0.03}
```

**部署方式**

| 方式 | 推理速度 | 模型大小 |
|------|----------|----------|
| PyTorch (原始) | ~10ms/张 | ~14MB |
| ONNX Runtime | ~8ms/张 | ~14MB |
| TensorRT FP16 (GPU) | ~3ms/张 | ~7MB |
| ONNX INT8 量化 | ~4ms/张 | ~4MB |

---

## 三、项目实施路线

| 阶段 | 内容 | 产出 |
|------|------|------|
| **P0 - 数据** | 收集 → 清洗 → 标签映射 → 增强 → 划分 | 数据集（~15,000张，三类均衡） |
| **P1 - 基线** | MobileNetV2 标准训练 + 评估 | Baseline Acc ≥ 92% |
| **P2 - 优化** | 超参调优 + FocalLoss + 增强迭代 | Acc ≥ 95% |
| **P3 - 鲁棒性** | 监控场景噪声测试 + 错误分析 | 混淆矩阵 + 错误案例 |
| **P4 - 部署** | ONNX 导出 + 推理脚本 + Demo | 可运行演示 |

---

## 四、常见问题与应对

| 问题 | 表现 | 解决方案 |
|------|------|----------|
| **三轮样本极少** | 三轮 recall 低 | 过采样 + 强增强；合成数据 |
| **监控画质差** | 远距离车辆模糊 | 数据集中混入实际监控帧；MotionBlur |
| **二轮/三轮混淆** | 侧面视角难区分 | 多视角训练 + 增加三轮特征增强 |
| **载货三轮/四轮** | 可见轮子数与实际不符 | 结合车身比例和结构联合判断 |
| **类别不均衡** | 训练偏向多数类 | WeightedLoss / 重采样 / FocalLoss |

---

## 五、最终交付物

1. ✅ 训练完成的 MobileNetV2 分类模型（.pth + .onnx）
2. ✅ 推理脚本（输入图片/视频 → 输出分类结果 + 置信度）
3. ✅ Confusion Matrix + 每类别 Precision/Recall/F1 报告
4. ✅ 目标指标：整体 Accuracy ≥ **95%**，三类 F1-score ≥ **0.90**

---

## 六、参考资源

- MobileNetV2: *Inverted Residuals and Linear Bottlenecks* (Sandler et al., 2018)
- BIT-Vehicle Dataset: 北京理工大学监控车辆数据集
- UA-DETRAC: 监控场景多目标检测与跟踪数据集
- PyTorch MobileNet 官方实现 / torchvision.models.mobilenet_v2
