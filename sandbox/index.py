import torch
import torch.nn as nn
import torchvision.models as models
import wandb

model_resnet18_pretrained = models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1)


model_resnet18_pretrained.fc = nn.Linear(in_features=512, out_features=10)
# print(model_resnet18_pretrained.fc)


resnet = models.resnet18(weights=None)
print(resnet.conv1)

resnet.conv1 = nn.Conv2d(
    in_channels=1,  # <-- MNIST has 1 channel
    out_channels=64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)

resnet.fc = nn.Linear(512, 10)  # 10 classes for MNIST


print(resnet.conv1)

print("-----------------------------------")

for param in resnet.parameters():
    print("shape: ", param.shape)
