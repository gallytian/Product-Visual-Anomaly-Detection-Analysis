"""
PaDiM 异常检测算法（论文复现）

论文: PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection
      and Localization (ICPR 2021)

流程:
1. 预训练 Wide ResNet-50-2 提取 layer1 / layer2 / layer3 特征，上采样拼接
2. 随机选择 d 个特征维度降维（论文: ResNet18 取 100 维, WRN50 取 550 维）
3. 对每个 patch 位置估计多元高斯分布（均值 + 完整协方差 + eps 正则化）
4. 测试时计算马氏距离作为异常分数
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from .backbone import WideResNetFeatureExtractor


class PaDiM:
    """PaDiM: 基于 patch 级高斯分布建模的无监督异常检测"""

    def __init__(
        self,
        layers=("layer1", "layer2", "layer3"),
        n_dims=100,
        eps=0.01,
        smooth_sigma=4.0,
        device="cpu",
        seed=42,
    ):
        """
        Args:
            layers: 提取特征的层名（论文: layer1 + layer2 + layer3）
            n_dims: 降维后的特征维度（论文 WRN50 用 550; CPU 适配默认 100）
            eps: 协方差正则化系数
            smooth_sigma: 图像级分数的异常图高斯平滑参数
            device: 运行设备
            seed: 随机种子
        """
        self.device = device
        self.layers = list(layers)
        self.n_dims = n_dims
        self.eps = eps
        self.smooth_sigma = smooth_sigma
        self.seed = seed

        torch.manual_seed(seed)
        self.backbone = WideResNetFeatureExtractor(self.layers, device=device)
        self.total_dim = sum(self.backbone.channels)
        self.dims_idx = torch.randperm(self.total_dim)[:n_dims]

        self.mean = None
        self.cov_inv = None
        self.grid = None
        self.fit_info = {}

    # ---------- 特征提取 ----------

    def _extract(self, images):
        """提取多尺度特征, 上采样拼接后降维为 (B, d, h, w)"""
        feats = self.backbone(images)
        target = feats[0].shape[-2:]
        ups = [
            F.interpolate(f, size=target, mode="bilinear", align_corners=False)
            for f in feats
        ]
        out = torch.cat(ups, dim=1)
        idx = self.dims_idx.to(out.device)
        return out[:, idx]

    # ---------- 训练: 每个 patch 位置的高斯分布估计 ----------

    def fit(self, dataloader):
        """遍历正常样本, 估计每个 patch 位置的均值与协方差"""
        feats_all = []
        for batch in dataloader:
            images = batch[0].to(self.device)
            f = self._extract(images)
            b, d, h, w = f.shape
            self.grid = (h, w)
            feats_all.append(
                f.permute(0, 2, 3, 1).reshape(b, h * w, d).cpu().numpy().astype(np.float32)
            )

        x = np.concatenate(feats_all, axis=0)  # (N, P, d)
        n, p, d = x.shape

        self.mean = x.mean(axis=0)  # (P, d)
        xc = torch.from_numpy(x - self.mean)
        cov = torch.einsum("npd,npe->pde", xc, xc) / (n - 1)
        cov = cov + self.eps * torch.eye(d)
        # 求逆用 float64 保证数值稳定, 存储转回 float32
        self.cov_inv = torch.linalg.inv(cov.double()).float().numpy()

        self.fit_info = {
            "num_images": int(n),
            "num_positions": int(p),
            "n_dims": int(d),
            "total_dim": int(self.total_dim),
        }
        return self.fit_info

    # ---------- 推理 ----------

    def predict(self, images):
        """预测一个 batch, 返回 (image_scores, anomaly_maps)"""
        images = images.to(self.device)
        b, _, hh, ww = images.shape

        f = self._extract(images)
        _, d, h, w = f.shape
        query = f.permute(0, 2, 3, 1).reshape(b, h * w, d).cpu().numpy()

        diff = query - self.mean[None]  # (B, P, d)
        mahal = np.einsum("bpd,pde,bpe->bp", diff, self.cov_inv, diff, optimize=True)

        amap = torch.from_numpy(mahal.astype(np.float32)).reshape(b, h, w).unsqueeze(1)
        amap = F.interpolate(amap, size=(hh, ww), mode="bilinear", align_corners=False)
        amap = amap.squeeze(1).numpy()

        image_scores = np.empty(b, dtype=np.float32)
        for i in range(b):
            image_scores[i] = gaussian_filter(amap[i], sigma=self.smooth_sigma).max()

        return image_scores, amap
