"""
MVTec AD 数据集加载器
"""

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    """MVTec AD 数据集: train 只有正常样本, test 含正常 + 各类缺陷"""

    def __init__(self, root, category, split="train", transform=None, resize=224):
        """
        Args:
            root: 数据集根目录
            category: 类别名称 (如 bottle, cable)
            split: 'train' 或 'test'
            transform: 图像变换（None 时使用 resize + ImageNet 归一化）
            resize: 图像预处理尺寸
        """
        self.root = Path(root)
        self.category = category
        self.split = split
        self.resize = resize

        self.transform = transform or T.Compose([
            T.Resize((resize, resize)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.samples = self._collect_samples()

    def _collect_samples(self):
        """收集样本路径: (图像, 掩码或 None, 标签)"""
        samples = []
        data_dir = self.root / self.category / self.split

        if self.split == "train":
            img_dir = data_dir / "good"
            for img_path in sorted(img_dir.glob("*")):
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    samples.append((img_path, None, 0))
        else:
            for defect_dir in sorted(data_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                is_normal = defect_dir.name == "good"
                label = 0 if is_normal else 1
                for img_path in sorted(defect_dir.glob("*")):
                    if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                        continue
                    mask_path = None
                    if not is_normal:
                        gt_dir = self.root / self.category / "ground_truth" / defect_dir.name
                        mask_path = gt_dir / (img_path.stem + "_mask" + img_path.suffix)
                    samples.append((img_path, mask_path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        h, w = image.shape[-2], image.shape[-1]
        mask = torch.zeros(1, h, w)
        if mask_path is not None and mask_path.exists():
            mask_img = Image.open(mask_path).convert("L")
            # 掩码用最近邻缩放, 避免插值模糊缺陷边界
            mask_img = mask_img.resize((w, h), Image.NEAREST)
            mask = (torch.from_numpy(np.asarray(mask_img, dtype=np.float32)) > 127.5).float().unsqueeze(0)

        return image, label, mask, str(img_path)


def get_dataloader(root, category, split, batch_size=8, num_workers=2, resize=224):
    """创建数据加载器"""
    dataset = MVTecDataset(root, category, split, resize=resize)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=False,
    )
