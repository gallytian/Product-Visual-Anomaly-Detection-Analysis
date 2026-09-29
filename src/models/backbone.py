"""
特征提取骨干网络

使用 torchvision 预训练 Wide ResNet-50-2 (ImageNet-1K),
支持按层名提取中间特征 —— 与 PatchCore / PaDiM 论文官方实现一致。
"""

import torch
import torchvision
from torchvision.models import Wide_ResNet50_2_Weights


class WideResNetFeatureExtractor:
    """Wide ResNet-50-2 多尺度特征提取器"""

    LAYER_NAMES = ("layer1", "layer2", "layer3", "layer4")
    LAYER_CHANNELS = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}

    def __init__(self, layers=("layer2", "layer3"), device="cpu"):
        """
        Args:
            layers: 需要提取的层名列表, 可选 layer1 ~ layer4
            device: 运行设备
        """
        self.layers = list(layers)
        self.device = device

        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
        model = torchvision.models.wide_resnet50_2(weights=weights)
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.channels = [self.LAYER_CHANNELS[name] for name in self.layers]

    def __call__(self, images):
        """前向到所需的最深层, 返回所选层的特征列表"""
        needed = set(self.layers)
        deepest = max(self.layers, key=self.LAYER_NAMES.index)

        with torch.no_grad():
            m = self.model
            x = m.conv1(images)
            x = m.bn1(x)
            x = m.relu(x)
            x = m.maxpool(x)

            feats = {}
            for name in self.LAYER_NAMES:
                x = getattr(m, name)(x)
                if name in needed:
                    feats[name] = x
                if name == deepest:
                    break
        return [feats[name] for name in self.layers]
