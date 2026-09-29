# Product Visual Anomaly Detection Analysis - Task Specification

## 项目概述

**Project Name:** Product Visual Anomaly Detection Analysis System based on MVTec AD  
**项目周期：** 3-4周  
**难度等级：** 中等偏上  
**Objective:** Build a complete product visual anomaly detection system, implement and compare multiple SOTA methods, and output experimental reports

---

## 一、项目背景

### 1.1 业务场景

工业制造中，产品表面缺陷检测是质量控制的核心环节。传统人工检测效率低、成本高，且容易漏检。本项目聚焦**开集异常检测**（Open-set Anomaly Detection）——只使用正常样本训练，检测训练时未见过的未知缺陷类型。

### 1.2 技术前沿

2025-2026年工业异常检测的核心趋势：
- 从"分类已知缺陷"走向"发现未知异常"
- 无监督/半监督方法成为主流
- 预训练视觉模型（DINOv2/CLIP）在工业场景的应用

### 1.3 项目价值

- 掌握工业异常检测的核心技术
- 理解无监督学习的实际应用场景
- 为简历增加一个前沿的CV项目

---

## 二、技术栈要求

### 2.1 必备知识

| 技能 | 要求程度 | 说明 |
|------|---------|------|
| Python | 熟练 | 核心开发语言 |
| PyTorch | 熟练 | 深度学习框架 |
| 计算机视觉基础 | 了解 | CNN、特征提取、数据增强 |
| 机器学习基础 | 了解 | 评估指标、模型对比 |

### 2.2 可选知识（加分项）

- 预训练模型（timm库）
- 异常检测相关论文阅读能力
- 实验报告撰写能力

---

## 三、项目任务分解

### 阶段1：环境搭建与数据准备（3-5天）

#### 任务1.1：环境配置
**目标：** 搭建完整的开发环境

**具体步骤：**
1. 创建Python虚拟环境（Python 3.9+）
2. 安装PyTorch（CUDA版本根据GPU选择）
3. 安装依赖包：
   ```bash
   pip install torch torchvision
   pip install timm scikit-learn pandas numpy matplotlib seaborn
   pip install opencv-python pillow tqdm pyyaml omegaconf
   pip install tensorboard wandb  # 可选：实验跟踪
   ```
4. 验证安装：
   ```python
   import torch
   print(f"PyTorch: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```

**验收标准：**
- [ ] 所有依赖包安装成功
- [ ] PyTorch能正常使用GPU（如果有）
- [ ] 能够运行示例代码

---

#### 任务1.2：数据集下载与理解
**目标：** 获取MVTec AD 2数据集，理解数据结构

**具体步骤：**
1. 申请MVTec AD 2数据集：
   - 访问：https://www.mvtec.com/company/research/datasets/mvtec-ad-2
   - 填写申请表单（需要学术/企业邮箱）
   - 等待1-3个工作日审核
   
2. 如果申请困难，使用替代方案：
   - MVTec AD（原版）：https://www.mvtec.com/company/research/datasets/mvtec-ad
   - VisA数据集：https://github.com/amazon-science/spot-diff

3. 下载并解压数据集到 `data/raw/mvtec_ad2/`

4. 理解数据结构：
   ```
   mvtec_ad2/
   ├── bottle/
   │   ├── train/good/          # 正常训练样本（约200张）
   │   ├── test/good/           # 正常测试样本
   │   ├── test/defect_type_1/  # 异常测试样本
   │   └── ground_truth/        # 异常区域mask
   ├── cable/
   ├── ...
   ```

**验收标准：**
- [ ] 数据集下载完成
- [ ] 能够用代码遍历数据集
- [ ] 理解train/test/ground_truth的含义

---

#### 任务1.3：数据探索与可视化
**目标：** 深入理解数据集特征

**具体步骤：**
1. 统计每个类别的样本数量
2. 分析图像尺寸分布
3. 可视化正常样本和异常样本的对比
4. 分析异常类型和mask特征

**输出：**
- 数据统计表格（CSV格式）
- 可视化图片（至少10张对比图）
- 数据探索notebook（Jupyter）

**验收标准：**
- [ ] 完成数据统计
- [ ] 生成可视化图片
- [ ] 能够清晰说明数据集特点

---

### 阶段2：核心模型实现（7-10天）

#### 任务2.1：PatchCore实现
**目标：** 实现PatchCore异常检测算法

**论文参考：** https://arxiv.org/abs/2106.08265

**核心思想：**
1. 使用预训练CNN（WideResNet50）提取多尺度特征
2. 构建正常样本的特征记忆库（Memory Bank）
3. 通过核心集采样（Coreset Sampling）压缩记忆库
4. 测试时计算样本与记忆库的最小距离作为异常分数

**实现要点：**
- 特征提取：使用timm库加载预训练模型
- 记忆库构建：收集所有正常样本的特征
- 核心集采样：贪心策略选择最具代表性的样本
- KNN检索：使用sklearn的NearestNeighbors
- 异常图生成：将patch级分数上采样到原图尺寸

**代码结构：**
```python
class PatchCore:
    def __init__(self, backbone, coreset_ratio, n_neighbors):
        # 初始化backbone和参数
    
    def extract_features(self, x):
        # 提取多尺度特征
    
    def fit(self, dataloader):
        # 构建记忆库
    
    def predict(self, x):
        # 预测异常分数和异常图
```

**验收标准：**
- [ ] 代码能够正常运行
- [ ] 在bottle类别上AUROC > 0.90
- [ ] 能够生成异常热力图

---

#### 任务2.2：PaDiM实现
**目标：** 实现PaDiM异常检测算法

**论文参考：** https://arxiv.org/abs/2011.08767

**核心思想：**
1. 提取每个patch位置的特征向量
2. 对每个位置建模多元高斯分布（均值+协方差）
3. 测试时计算马氏距离

**实现要点：**
- 特征提取：同PatchCore
- 高斯建模：对每个patch位置计算均值和协方差
- 马氏距离：计算测试样本与高斯分布的距离
- 降维：随机选择部分维度（如100维）减少计算量

**验收标准：**
- [ ] 代码能够正常运行
- [ ] 在bottle类别上AUROC > 0.85
- [ ] 能够生成异常热力图

---

#### 任务2.3：STFPM实现（可选）
**目标：** 实现STFPM异常检测算法

**论文参考：** https://arxiv.org/abs/2103.04257

**核心思想：**
1. 教师网络：预训练CNN，冻结参数
2. 学生网络：相同架构，从头训练
3. 学生只学习正常样本的特征分布
4. 测试时比较教师和学生的特征差异

**实现要点：**
- 需要训练学生网络（约100个epoch）
- 特征匹配损失：MSE loss
- 多层特征融合

**验收标准：**
- [ ] 代码能够正常运行
- [ ] 在bottle类别上AUROC > 0.85

---

#### 任务2.4：FastFlow实现（可选，难度较高）
**目标：** 实现FastFlow异常检测算法

**论文参考：** https://arxiv.org/abs/2111.07677

**核心思想：**
使用2D Normalizing Flow学习正常样本的特征分布

**实现建议：**
- 可以使用FrEIA库实现Normalizing Flow
- 或者简化为自编码器版本

**验收标准：**
- [ ] 代码能够正常运行
- [ ] 在bottle类别上AUROC > 0.80

---

### 阶段3：评估与对比（3-5天）

#### 任务3.1：评估指标实现
**目标：** 实现工业异常检测的标准评估指标

**需要实现的指标：**
1. **Image AUROC**：图像级异常检测的AUROC
2. **Pixel AUROC**：像素级异常定位的AUROC
3. **AUPRO**：Per-Region Overlap（工业标准指标）
4. **F1-max**：最佳F1分数

**实现要点：**
- 使用sklearn的roc_auc_score
- AUPRO需要连通域分析（可参考官方实现）
- 阈值选择：基于F1-max的阈值

**验收标准：**
- [ ] 所有指标能够正确计算
- [ ] 与论文报告的数值接近（±5%）

---

#### 任务3.2：多类别实验
**目标：** 在所有类别上运行实验

**实验设置：**
- 选择至少5个类别（如：bottle, cable, capsule, grid, hazelnut）
- 每个类别独立训练和测试
- 记录所有指标

**输出：**
- 结果表格（CSV格式）
- 包含：类别、模型、Image-AUROC、Pixel-AUROC、F1-max

**验收标准：**
- [ ] 完成至少5个类别的实验
- [ ] 结果表格完整

---

#### 任务3.3：方法对比分析
**目标：** 对比不同方法的优劣

**分析维度：**
1. 检测性能（AUROC、F1）
2. 定位精度（Pixel-AUROC、AUPRO）
3. 计算效率（训练时间、推理时间）
4. 内存占用

**输出：**
- 对比表格
- 可视化图表（柱状图、折线图）
- 分析结论（哪种方法在什么场景下更好）

**验收标准：**
- [ ] 完成方法对比
- [ ] 生成可视化图表
- [ ] 有清晰的结论

---

### 阶段4：可视化与报告（3-5天）

#### 任务4.1：异常可视化
**目标：** 生成直观的异常检测结果可视化

**可视化内容：**
1. 输入图像 vs 异常热力图 vs Ground Truth
2. 正常样本 vs 异常样本的对比
3. 不同方法的异常图对比
4. 失败案例分析（误检、漏检）

**实现要点：**
- 使用matplotlib生成可视化
- 异常图使用jet colormap
- 叠加原图和异常图（alpha=0.5）

**输出：**
- 至少20张可视化图片
- 按类别和缺陷类型组织

**验收标准：**
- [ ] 可视化图片清晰直观
- [ ] 包含正常和异常样本
- [ ] 包含失败案例分析

---

#### 任务4.2：实验报告撰写
**目标：** 撰写完整的实验报告

**报告结构：**
```
1. 项目概述
   - 背景和目标
   - 技术路线

2. 数据集
   - 数据集介绍
   - 数据统计
   - 数据特点

3. 方法介绍
   - PatchCore原理
   - PaDiM原理
   - 其他方法（如有）

4. 实验设置
   - 实现细节
   - 超参数设置
   - 评估指标

5. 实验结果
   - 定量结果（表格）
   - 定性结果（可视化）
   - 方法对比

6. 分析与讨论
   - 各方法的优劣
   - 失败案例分析
   - 改进方向

7. 总结与展望
```

**报告要求：**
- 字数：3000-5000字
- 图表：至少10张
- 参考文献：至少5篇

**验收标准：**
- [ ] 报告结构完整
- [ ] 图表清晰美观
- [ ] 分析有深度

---

#### 任务4.3：代码整理与文档
**目标：** 整理代码，编写文档

**具体步骤：**
1. 代码重构：
   - 模块化设计
   - 添加注释
   - 统一代码风格

2. README编写：
   - 项目介绍
   - 安装指南
   - 使用说明
   - 结果展示

3. 代码开源（可选）：
   - 上传到GitHub
   - 添加LICENSE

**验收标准：**
- [ ] 代码结构清晰
- [ ] README完整
- [ ] 他人能够复现结果

---

## 四、时间规划建议

| 阶段 | 任务 | 建议时间 | 累计时间 |
|------|------|---------|---------|
| 阶段1 | 环境搭建与数据准备 | 3-5天 | 5天 |
| 阶段2 | 核心模型实现 | 7-10天 | 15天 |
| 阶段3 | 评估与对比 | 3-5天 | 20天 |
| 阶段4 | 可视化与报告 | 3-5天 | 25天 |

**总计：约25天（3.5周）**

---

## 五、参考资料

### 5.1 论文

1. **PatchCore**: Towards Total Recall in Industrial Anomaly Detection (CVPR 2022)
   - https://arxiv.org/abs/2106.08265
   - 官方代码：https://github.com/amazon-science/patchcore-inspection

2. **PaDiM**: PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection (ICPR 2021)
   - https://arxiv.org/abs/2011.08767
   - 官方代码：https://github.com/xiahaifeng1995/PaDiM-anomaly-detection-localization

3. **STFPM**: Student-Teacher Feature Pyramid Matching for Anomaly Detection (CVPR 2021)
   - https://arxiv.org/abs/2103.04257
   - 官方代码：https://github.com/defectDetection/STFPM

4. **FastFlow**: FastFlow: Unsupervised Anomaly Detection and Localization with 2D Normalizing Flows (WACV 2022)
   - https://arxiv.org/abs/2111.07677
   - 官方代码：https://github.com/gathierry/FastFlow

### 5.2 数据集

1. **MVTec AD 2**: https://www.mvtec.com/company/research/datasets/mvtec-ad-2
2. **MVTec AD**: https://www.mvtec.com/company/research/datasets/mvtec-ad
3. **VisA**: https://github.com/amazon-science/spot-diff

### 5.3 开源项目参考

1. **Anomaly Detection Toolbox**: https://github.com/open-mmlab/mmagic
2. **MVTec AD Benchmark**: https://github.com/taewookkim/DeepAnomalyDetection

### 5.4 技术博客

1. 工业异常检测综述：https://blog.csdn.net/m0_63774211/article/details/155980978
2. PatchCore详解：https://zhuanlan.zhihu.com/p/xxx（搜索"PatchCore 工业异常检测"）

---

## 六、常见问题

### Q1: 没有GPU怎么办？
**A:** 可以使用CPU运行，但速度会慢很多。建议使用Google Colab（免费GPU）或租用云GPU。

### Q2: MVTec AD 2申请不下来怎么办？
**A:** 使用MVTec AD（原版）或VisA数据集替代，方法相同。

### Q3: 实现的效果达不到论文水平怎么办？
**A:** 
- 检查数据预处理是否正确
- 检查超参数设置
- 参考官方代码的实现细节
- 允许±5%的性能差距

### Q4: 如何验证实现的正确性？
**A:** 
- 在bottle类别上验证（最简单）
- 对比官方代码的结果
- 可视化异常图，检查是否合理

---

## 七、交付物清单

### 必须交付：
- [ ] 完整的代码（包含PatchCore和PaDiM）
- [ ] 数据探索notebook
- [ ] 实验结果表格（CSV）
- [ ] 可视化图片（至少20张）
- [ ] 实验报告（PDF或Markdown）
- [ ] README文档

### 可选交付：
- [ ] STFPM/FastFlow实现
- [ ] 更多类别的实验结果
- [ ] 代码开源到GitHub
- [ ] 技术博客文章

---


## 附录：代码模板

### A.1 数据集加载器模板

```python
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class MVTecDataset(Dataset):
    def __init__(self, root, category, split='train', transform=None):
        self.root = Path(root)
        self.category = category
        self.split = split
        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        self.samples = self._collect_samples()
    
    def _collect_samples(self):
        samples = []
        data_dir = self.root / self.category / self.split
        
        if self.split == 'train':
            img_dir = data_dir / 'good'
            for img_path in sorted(img_dir.glob('*')):
                samples.append((img_path, None, 0))
        else:
            for defect_type in sorted(data_dir.iterdir()):
                if not defect_type.is_dir():
                    continue
                is_normal = defect_type.name == 'good'
                label = 0 if is_normal else 1
                
                for img_path in sorted(defect_type.glob('*')):
                    mask_path = None
                    if not is_normal:
                        gt_dir = self.root / self.category / 'ground_truth' / defect_type.name
                        mask_name = img_path.stem + '_mask' + img_path.suffix
                        mask_path = gt_dir / mask_name
                    samples.append((img_path, mask_path, label))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        mask = torch.zeros(1, image.shape[-2], image.shape[-1])
        if mask_path and mask_path.exists():
            mask_img = Image.open(mask_path).convert('L')
            mask_img = T.Resize((image.shape[-2], image.shape[-1]))(mask_img)
            mask = (T.ToTensor()(mask_img) > 0.5).float()
        
        return image, label, mask, str(img_path)
```

### A.2 评估指标模板

```python
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve


def compute_metrics(scores, labels, anomaly_maps=None, masks=None):
    metrics = {}
    
    # 图像级AUROC
    metrics['image_auroc'] = roc_auc_score(labels, scores)
    
    # 最佳F1
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    metrics['f1_max'] = np.max(f1_scores)
    
    # 像素级AUROC
    if anomaly_maps is not None and masks is not None:
        metrics['pixel_auroc'] = roc_auc_score(
            masks.flatten(), 
            anomaly_maps.flatten()
        )
    
    return metrics
```

---

**祝项目顺利！**
