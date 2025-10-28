
# fmt: off
# isort: skip_file
# DO NOT reorganize imports - warnings filter must be FIRST!

import warnings
import os

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import torch
import torch.nn as nn
import timm 
from timm import create_model

# fmt: on

pvt_models = timm.list_models('*pvt*', pretrained=True)


for i, model in enumerate(pvt_models):
    print(f"      - {model}")


# PVT backbone
backbone = timm.create_model(
    'pvt_v2_b2.in1k', pretrained=True, features_only=True)
# Channels from timm summary: [64, 128, 320, 512]
channels = [64, 128, 320, 512]

H, W = 224, 224
x = torch.randn(1, 3, H, W)
features = backbone(x)
for c in features:
    print(c.shape)

exit()


class FPN(nn.Module):
    def __init__(self, channels, out_channels=256):
        super().__init__()
        # 1x1 lateral layers
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in channels])
        # 3x3 smooth layers
        self.smooth = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in channels])

    def forward(self, features):
        C1, C2, C3, C4 = features

        P4 = self.lateral[3](C4)
        P3 = self.lateral[2](C3) + nn.functional.interpolate(P4,
                                                             size=C3.shape[-2:], mode='nearest')
        P2 = self.lateral[1](C2) + nn.functional.interpolate(P3,
                                                             size=C2.shape[-2:], mode='nearest')
        P1 = self.lateral[0](C1) + nn.functional.interpolate(P2,
                                                             size=C1.shape[-2:], mode='nearest')

        P1 = self.smooth[0](P1)
        P2 = self.smooth[1](P2)
        P3 = self.smooth[2](P3)
        P4 = self.smooth[3](P4)
        return [P1, P2, P3, P4]


# Example
x = torch.randn(1, 3, 224, 224)
features = backbone(x)
fpn = FPN(channels)
pyramid = fpn(features)
for i, p in enumerate(pyramid):
    print(f"P{i+1} shape:", p.shape)

# **Output: **


# P1 shape: torch.Size([1, 256, 56, 56])
# P2 shape: torch.Size([1, 256, 28, 28])
# P3 shape: torch.Size([1, 256, 14, 14])
# P4 shape: torch.Size([1, 256, 7, 7])
# ```

# ✅ Each stage has ** same channel dimension(256)**,
# but different spatial scales — perfect for multi-scale prediction heads.
