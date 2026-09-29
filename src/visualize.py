"""
可视化模块: 异常热力图 / GT 掩码 / 叠加图
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CMAP = matplotlib.colormaps["jet"]


def _to_display_map(amap, size_wh):
    """把异常图缩放到原图尺寸并归一化到 [0,1]"""
    amap = np.asarray(amap, dtype=np.float32).squeeze()
    img = Image.fromarray(amap).resize(size_wh, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def visualize_results(image_paths, anomaly_maps, masks, labels, output_dir, n_samples=8, dpi=110):
    """
    为若干样本生成 原图/异常热力图/GT/叠加 四联图并保存

    Args:
        image_paths: 图像路径列表
        anomaly_maps: 异常图 (N, H, W)，尺寸为预处理尺寸
        masks: GT 掩码 (N, H, W) 或 (N, 1, H, W)
        labels: 图像级标签
        output_dir: 输出目录
        n_samples: 生成数量
        dpi: 输出分辨率
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = min(n_samples, len(image_paths))
    for i in range(n):
        img = Image.open(image_paths[i]).convert("RGB")
        w, h = img.size
        arr = np.asarray(img, dtype=np.float32) / 255.0

        heat = _to_display_map(anomaly_maps[i], (w, h))
        heat_rgb = CMAP(heat)[..., :3]

        mask = np.asarray(masks[i], dtype=np.float32).squeeze()
        mask_img = Image.fromarray(((mask > 0.5).astype(np.uint8) * 255)).resize((w, h), Image.NEAREST)
        mask_show = np.asarray(mask_img, dtype=np.float32) / 255.0

        overlay = np.clip(0.55 * arr + 0.45 * heat_rgb, 0, 1)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), dpi=dpi)
        axes[0].imshow(arr)
        axes[0].set_title("Input", fontsize=10)
        axes[1].imshow(heat_rgb)
        axes[1].set_title("Anomaly Map", fontsize=10)
        axes[2].imshow(mask_show, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Ground Truth", fontsize=10)
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay", fontsize=10)
        for ax in axes:
            ax.axis("off")

        tag = "Normal" if int(labels[i]) == 0 else "Anomaly"
        fig.suptitle(
            f"{Path(image_paths[i]).parent.name} / {Path(image_paths[i]).name}  [{tag}]",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"sample_{i + 1:03d}.png", bbox_inches="tight")
        plt.close(fig)
