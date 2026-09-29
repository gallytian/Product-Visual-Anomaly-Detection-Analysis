"""
PatchCore 异常检测算法（论文复现 + CPU 适配）

论文: Towards Total Recall in Industrial Anomaly Detection (CVPR 2022)

流程:
1. 预训练 Wide ResNet-50-2 提取 layer2 / layer3 特征，上采样拼接
2. 3x3 局部邻域平均池化，聚合局部上下文
3. 随机投影 + 贪心核心集采样，压缩记忆库
4. 测试时 KNN 计算最近邻距离，并重加权（近邻密集时降权）
5. 异常图上采样 + 高斯平滑，得到图像级分数与像素级异常图
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors

from .backbone import WideResNetFeatureExtractor


class PatchCore:
    """PatchCore: 基于特征记忆库的无监督异常检测"""

    def __init__(
        self,
        layers=("layer2", "layer3"),
        coreset_ratio=0.1,
        n_neighbors=9,
        max_bank_size=40000,
        projection_dim=128,
        smooth_sigma=4.0,
        device="cpu",
        seed=42,
    ):
        """
        Args:
            layers: 提取特征的层名（论文: layer2 + layer3）
            coreset_ratio: 核心集采样比例
            n_neighbors: 重加权使用的近邻数
            max_bank_size: 记忆库规模上限（CPU 适配: 贪心采样前随机截断）
            projection_dim: 核心集采样的随机投影维度
            smooth_sigma: 图像级分数的异常图高斯平滑参数
            device: 运行设备
            seed: 随机种子
        """
        self.device = device
        self.layers = list(layers)
        self.coreset_ratio = coreset_ratio
        self.n_neighbors = n_neighbors
        self.max_bank_size = max_bank_size
        self.projection_dim = projection_dim
        self.smooth_sigma = smooth_sigma
        self.seed = seed

        torch.manual_seed(seed)
        self.backbone = WideResNetFeatureExtractor(self.layers, device=device)

        self.memory_bank = None
        self.knn = None
        self.fit_info = {}

    # ---------- 特征提取 ----------

    def _extract(self, images):
        """提取多尺度特征并拼接为 (B, C, h, w)"""
        feats = self.backbone(images)
        target = feats[0].shape[-2:]
        ups = [
            F.interpolate(f, size=target, mode="bilinear", align_corners=False)
            for f in feats
        ]
        return torch.cat(ups, dim=1)

    # ---------- 训练: 构建记忆库 ----------

    def fit(self, dataloader):
        """遍历正常样本, 构建压缩后的特征记忆库"""
        feats_all = []
        for batch in dataloader:
            images = batch[0].to(self.device)
            f = self._extract(images)
            f = F.avg_pool2d(f, 3, stride=1, padding=1)  # 局部邻域聚合
            b, c, h, w = f.shape
            feats_all.append(
                f.permute(0, 2, 3, 1).reshape(b * h * w, c).cpu().numpy().astype(np.float32)
            )

        bank = np.concatenate(feats_all, axis=0)
        n_all = len(bank)

        rng = np.random.default_rng(self.seed)
        if n_all > self.max_bank_size:
            keep = rng.choice(n_all, size=self.max_bank_size, replace=False)
            bank = bank[np.sort(keep)]

        core_idx = self._greedy_coreset(bank, rng)
        self.memory_bank = bank[core_idx]
        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1, algorithm="brute", n_jobs=-1
        )
        self.knn.fit(self.memory_bank)

        self.fit_info = {
            "patch_total": int(n_all),
            "bank_used": int(len(bank)),
            "memory_bank": int(len(self.memory_bank)),
            "feature_dim": int(self.memory_bank.shape[1]),
        }
        return self.fit_info

    def _greedy_coreset(self, bank, rng):
        """随机投影后的贪心核心集采样 (k-center greedy)"""
        n = len(bank)
        target = max(1, int(round(n * self.coreset_ratio)))
        if target >= n:
            return np.arange(n)

        d = bank.shape[1]
        proj = rng.normal(size=(d, self.projection_dim)).astype(np.float32) / np.sqrt(d)
        feat = bank @ proj  # (n, projection_dim), 降低贪心采样的计算量

        selected = np.empty(target, dtype=np.int64)
        selected[0] = rng.integers(n)
        min_dist = np.linalg.norm(feat - feat[selected[0]], axis=1)
        for i in range(1, target):
            nxt = int(np.argmax(min_dist))
            selected[i] = nxt
            dist = np.linalg.norm(feat - feat[nxt], axis=1)
            np.minimum(min_dist, dist, out=min_dist)
        return selected

    # ---------- 推理 ----------

    def predict(self, images):
        """预测一个 batch, 返回 (image_scores, anomaly_maps)"""
        images = images.to(self.device)
        b, _, hh, ww = images.shape

        f = self._extract(images)
        f = F.avg_pool2d(f, 3, stride=1, padding=1)
        _, c, h, w = f.shape
        query = (
            f.permute(0, 2, 3, 1).reshape(b * h * w, c).cpu().numpy().astype(np.float32)
        )

        dists, _ = self.knn.kneighbors(query, n_neighbors=self.n_neighbors + 1)
        d0 = dists[:, 0]
        # 重加权: k 个近邻距离都相近说明记忆库中该处密集, 降低置信度
        ratio = np.exp(d0 - logsumexp(dists, axis=1))
        weights = 1.0 - ratio
        patch_scores = (weights * d0).astype(np.float32)

        amap = torch.from_numpy(patch_scores.reshape(b, h, w)).unsqueeze(1)
        amap = F.interpolate(amap, size=(hh, ww), mode="bilinear", align_corners=False)
        amap = amap.squeeze(1).numpy()

        image_scores = np.empty(b, dtype=np.float32)
        for i in range(b):
            image_scores[i] = gaussian_filter(amap[i], sigma=self.smooth_sigma).max()

        return image_scores, amap
