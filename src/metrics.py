"""
评估指标模块

指标定义:
- Image AUROC: 图像级异常分类的 ROC 曲线下面积
- Pixel AUROC: 像素级异常定位的 ROC 曲线下面积
- AUPRO: Per-Region Overlap 曲线下面积（区域级定位指标, FPR 上限 0.3）
- F1-max: 图像级最优阈值下的 F1 分数
"""

import numpy as np
from scipy.ndimage import label as connected_label
from sklearn.metrics import precision_recall_curve, roc_auc_score


def compute_metrics(image_scores, labels, anomaly_maps=None, masks=None, max_fpr=0.3):
    """
    计算全部评估指标

    Args:
        image_scores: 图像级异常分数 (N,)
        labels: 图像级标签 (N,) 0=正常, 1=异常
        anomaly_maps: 像素级异常图 (N, H, W)
        masks: 像素级 GT 掩码 (N, H, W) 或 (N, 1, H, W)
        max_fpr: AUPRO 的 FPR 积分上限

    Returns:
        dict: 指标字典
    """
    scores = np.asarray(image_scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)

    metrics = {
        "image_auroc": float(roc_auc_score(labels, scores)),
    }

    # F1-max（图像级, 在最优阈值处）
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-8, None)
    k = int(np.argmax(f1))
    metrics["f1_max"] = float(f1[k])
    if thresholds.size and 0 <= k < thresholds.size:
        metrics["f1_threshold"] = float(thresholds[k])

    # 像素级指标
    if anomaly_maps is not None and masks is not None:
        maps = np.asarray(anomaly_maps, dtype=np.float32)
        gts = np.asarray(masks, dtype=np.float32)
        if gts.ndim == 4:
            gts = gts[:, 0]

        metrics["pixel_auroc"] = float(
            roc_auc_score((gts > 0.5).ravel().astype(int), maps.ravel())
        )

        normal = labels == 0
        abnormal = labels == 1
        if abnormal.sum() > 0 and normal.sum() > 0:
            metrics["aupro"] = float(
                compute_aupro(maps[abnormal], gts[abnormal], maps[normal], max_fpr=max_fpr)
            )

    return metrics


def compute_aupro(anomaly_maps, masks, normal_maps, max_fpr=0.3, n_points=300):
    """
    计算 AUPRO (Per-Region Overlap)

    做法:
    1. 用正常图像的像素分数分布确定分割阈值, 得到 FPR 网格
    2. 每个阈值下计算每个缺陷连通域被检出的像素比例 (overlap)
    3. 对所有连通域取平均得到 PRO 值, 在 FPR ∈ [0, max_fpr] 上积分并归一化

    Args:
        anomaly_maps: 异常图像的异常图 (N, H, W)
        masks: 异常图像的 GT 掩码 (N, H, W)
        normal_maps: 正常图像的异常图 (M, H, W)
        max_fpr: FPR 积分上限
        n_points: FPR 采样点数

    Returns:
        float: AUPRO 值
    """
    # 1. 提取全部缺陷连通域的像素分数（升序）
    comps = []
    for amap, gt in zip(anomaly_maps, masks):
        lab, num = connected_label(gt > 0.5)
        for i in range(1, num + 1):
            px = amap[lab == i]
            if px.size > 0:
                comps.append(np.sort(px.astype(np.float64)))
    if not comps:
        return float("nan")

    # 2. 正常像素分数分布（升序）, 用于确定阈值
    normal_scores = np.sort(
        np.concatenate([m.ravel() for m in normal_maps]).astype(np.float64)
    )
    n_norm = normal_scores.size
    if n_norm == 0:
        return float("nan")

    # 3. 按 FPR 网格求阈值与 PRO
    fprs = np.linspace(0.0, max_fpr, n_points)
    pro = np.empty(n_points, dtype=np.float64)
    for i, fpr in enumerate(fprs):
        rank = n_norm - 1 - int(fpr * n_norm)
        rank = min(max(rank, 0), n_norm - 1)
        thr = normal_scores[rank]
        overlaps = [
            (c.size - np.searchsorted(c, thr, side="right")) / c.size for c in comps
        ]
        pro[i] = float(np.mean(overlaps))

    # 4. 积分（归一化到 FPR 区间）
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(pro, fprs) / max_fpr)
