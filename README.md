# Product Visual Anomaly Detection Analysis

A product visual anomaly detection and analysis system based on the MVTec AD dataset, implementing and comparing two mainstream unsupervised anomaly detection methods (PatchCore / PaDiM).

> **Background**: In product quality inspection, anomalous samples are scarce and their types are unpredictable, making supervised classification training infeasible. This project adopts an **open-set anomaly detection** approach — modeling only with normal samples to detect defect types never seen during training, while providing pixel-level defect localization.

---

## 1. 环境要求

| 项目 | 版本 / 说明 |
|------|-------------|
| Python | 3.9+（本项目在 3.12 上验证） |
| PyTorch | 2.0+（CPU 版本亦可运行） |
| 硬件 | CPU 可运行；有 NVIDIA GPU 时自动使用 |
| 显存 / 内存 | CPU 模式建议 ≥ 8GB 内存 |

本项目在**无 GPU 的 CPU 环境**下完成全部实验（5 类别 × 2 模型，约 21 分钟）。

---

## 2. 项目结构

```
工业检测项目/
├── src/                          # 核心代码
│   ├── dataset.py                # MVTec AD 数据加载器
│   ├── metrics.py                # 评估指标（Image/Pixel AUROC、AUPRO、F1-max）
│   ├── visualize.py              # 异常热力图可视化
│   └── models/
│       ├── backbone.py           # Wide ResNet-50-2 多尺度特征提取器
│       ├── patchcore.py          # PatchCore 算法
│       └── padim.py              # PaDiM 算法
├── scripts/                      # 可执行脚本
│   ├── train.py                  # 单类别实验
│   ├── run_all.py                # 批量实验 + 结果汇总
│   ├── explore_dataset.py        # 数据集统计探索
│   └── visualize_dataset.py      # 数据集可视化（8 类图表）
├── notebooks/
│   └── 01_data_exploration.ipynb # 数据探索 notebook
├── configs/
│   └── default.yaml              # 实验配置（模型超参数、实验类别）
├── data/raw/
│   └── mvtec_anomaly_detection/  # 数据集（需自行下载）
├── results/                      # 实验输出
│   ├── <model>/<category>/       # 各类别：metrics.json、scores.npz、visualizations/
│   ├── visualization/            # 数据探索图表（8 张）
│   ├── analysis/                 # 方法对比图、失败案例分析
│   ├── summary.csv               # 结果汇总表
│   └── summary.md                # 结果汇总报告
├── reports/                      # 实验报告
│   └── experiment_report.md      # 完整实验报告
├── assets/mathjax/               # 本地 MathJax（HTML 文档公式渲染）
├── index.html                    # 项目文档（含全部技术细节）
├── requirements.txt
└── PROJECT_TASK.md               # 项目任务书
```

---

## 3. 安装

```bash
# 建议使用虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 如需 GPU 版本 PyTorch（CPU 环境跳过此步）
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

验证安装：

```python
import torch, torchvision
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

---

## 4. 数据准备

下载 MVTec AD 数据集（约 4.9GB）：

- 官方地址：https://www.mvtec.com/company/research/datasets/mvtec-ad
- 无需申请，直接下载

解压到 `data/raw/`，确保目录结构如下：

```
data/raw/mvtec_anomaly_detection/
├── bottle/
│   ├── train/good/               # 正常训练样本（仅此目录参与建模）
│   ├── test/good/                # 正常测试样本
│   ├── test/<defect_type>/       # 各缺陷类型的测试样本
│   └── ground_truth/<defect_type>/  # 像素级 GT 掩码（*_mask.png）
├── cable/
└── ...（共 15 类）
```

---

## 5. 使用方法

### 5.1 数据探索

```bash
python scripts/explore_dataset.py      # 统计各类别样本数、缺陷类型
python scripts/visualize_dataset.py    # 生成 8 张数据探索图表
```

输出：`results/dataset_stats.json`、`results/visualization/*.png`

### 5.2 单类别实验

```bash
# PatchCore
python scripts/train.py --model patchcore --category bottle

# PaDiM
python scripts/train.py --model padim --category bottle

# 指定配置文件
python scripts/train.py --model patchcore --category cable --config configs/default.yaml
```

### 5.3 批量实验（多类别 × 多模型）

```bash
# 使用 configs/default.yaml 中配置的类别与模型
python scripts/run_all.py

# 自定义范围
python scripts/run_all.py --categories bottle cable carpet
python scripts/run_all.py --models patchcore
```

输出：
- `results/<model>/<category>/metrics.json` —— 该类别全部指标与耗时
- `results/<model>/<category>/scores.npz` —— 图像分数、标签、异常图（供后续分析）
- `results/<model>/<category>/visualizations/` —— 异常热力图四联图
- `results/summary.csv` / `summary.md` —— 汇总表

### 5.4 方法对比与失败案例分析

```bash
python scripts/analyze_results.py
```

输出：`results/analysis/` 下的对比柱状图与误检/漏检案例图

---

## 6. 实验结果

### 6.1 总体结果（5 类别平均）

| 模型 | Image AUROC | Pixel AUROC | AUPRO | F1-max | 平均建模耗时 |
|------|-------------|-------------|-------|--------|-------------|
| **PatchCore** | **0.9976** | **0.9845** | **0.9778** | **0.9889** | 131 s |
| PaDiM | 0.9520 | 0.9739 | 0.9411 | 0.9288 | 47 s |

### 6.2 各类别明细

| 模型 | 类别 | Image AUROC | Pixel AUROC | AUPRO | F1-max |
|------|------|-------------|-------------|-------|--------|
| PatchCore | bottle | 1.0000 | 0.9853 | 0.9982 | 1.0000 |
| PatchCore | cable | 0.9912 | 0.9856 | 0.9826 | 0.9681 |
| PatchCore | carpet | 0.9976 | 0.9900 | 0.9894 | 0.9888 |
| PatchCore | hazelnut | 1.0000 | 0.9869 | 0.9328 | 1.0000 |
| PatchCore | transistor | 0.9992 | 0.9744 | 0.9857 | 0.9877 |
| PaDiM | bottle | 1.0000 | 0.9832 | 0.9959 | 1.0000 |
| PaDiM | cable | 0.8606 | 0.9522 | 0.8987 | 0.8286 |
| PaDiM | carpet | 0.9952 | 0.9830 | 0.9766 | 0.9944 |
| PaDiM | hazelnut | 0.9550 | 0.9761 | 0.8680 | 0.9504 |
| PaDiM | transistor | 0.9492 | 0.9749 | 0.9664 | 0.8706 |

### 6.3 结论

- **PatchCore 全面领先**：四项指标均达「优秀」档，平均 Image AUROC 99.76%，与论文报告水平（约 99.6%）一致
- **PaDiM 短板在复杂结构**：cable 类别 Image AUROC 仅 86.06%，主因是 CPU 适配将特征维度从论文的 550 维降至 100 维
- **速度与精度权衡**：PaDiM 建模快约 3 倍（47s vs 131s），适合训练时间敏感的场景
- **定位能力接近**：两者 Pixel AUROC 差距仅 1.1 个百分点，主要差距在图像级判别

---

## 7. 评估指标

| 指标 | 层面 | 说明 |
|------|------|------|
| **Image AUROC** | 图像级 | 能否发现异常图（阈值无关），最重要 |
| **Pixel AUROC** | 像素级 | 异常定位的像素级精度 |
| **AUPRO** | 区域级 | 每个缺陷连通域被整体找出的比例（FPR ≤ 0.3） |
| **F1-max** | 图像级 | 最优阈值下的精确率/召回率平衡 |

优良参考区间：Image AUROC ≥ 99% 优秀 / 97~99% 良好 / 95~97% 可用。

---

## 8. 算法实现要点

### PatchCore（CVPR 2022）

1. WRN-50-2 提取 layer2 + layer3 特征，上采样拼接（1536 维）
2. 3×3 局部邻域平均池化聚合上下文
3. 随机投影（128 维）+ 贪心 k-center 核心集采样，压缩记忆库至 10%
4. 测试时 KNN（K=9）检索 + 近邻密度重加权
5. 异常图上采样 + 高斯平滑，图像级分数取最大值

### PaDiM（ICPR 2021）

1. WRN-50-2 提取 layer1 + layer2 + layer3 特征，上采样拼接（1792 维）
2. 随机降维至 100 维（论文 WRN50 为 550 维，此处受 CPU 内存限制）
3. 每个 patch 位置估计多元高斯分布（均值 + 完整协方差 + ε 正则化）
4. 测试时计算马氏距离作为异常分数

---

## 9. 复现说明

完整复现流程：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据集到 data/raw/mvtec_anomaly_detection/

# 3. 数据探索（可选）
python scripts/explore_dataset.py
python scripts/visualize_dataset.py

# 4. 批量实验（5 类别 × 2 模型）
python scripts/run_all.py

# 5. 结果分析与图表
python scripts/analyze_results.py
```

随机种子固定为 42（见 `configs/default.yaml`），结果可复现。

---

## 10. 参考文献

1. Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). **MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection**. CVPR 2019.
2. Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P. (2022). **Towards Total Recall in Industrial Anomaly Detection**. CVPR 2022. (PatchCore)
3. Defard, T., Setkov, A., Loesch, A., & Audigier, R. (2021). **PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization**. ICPR 2021.
4. Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2020). **Uninformed Students: Student-Teacher Anomaly Detection with Discriminative Latent Embeddings**. CVPR 2020. (STFPM 相关)
5. Yu, J., Zheng, Y., Wang, X., Li, W., Wu, Y., Zhao, R., & Wu, L. (2021). **FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows**. WACV 2022.

---

## 11. 许可

数据集遵循 MVTec 官方许可（CC BY-NC-SA 4.0）。本项目代码仅用于学习研究。
