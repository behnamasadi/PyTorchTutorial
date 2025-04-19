import torch
import torch.nn as nn
import torchvision.models as models


class ResNetMLP(nn.Module):
    def __init__(self):
        super(ResNetMLP, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,  # <-- MNIST has 1 channel
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.resnet.fc = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = self.resnet(x)
        return x
