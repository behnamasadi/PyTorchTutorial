from torchinfo import summary  # pip install torchinfo
import torch
import torch.nn as nn
import torchvision.models as models


model_vgg19_bn = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
for param in model_vgg19_bn.features.parameters():
    param.requires_grad = False

print(model_vgg19_bn)
# # print(model_vgg19_bn.features.parameters())

# for param in model_vgg19_bn.features.parameters():
#     print(param.shape)


# print(model_vgg19_bn)


# exit()
# # Freeze convolutional layers
# for param in model_vgg19_bn.features.parameters():
#     param.requires_grad = False

# print("model_vgg19_bn.classifier", model_vgg19_bn.classifier)


# resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# print(resnet18)


# print("resnet18 input size: ", resnet18.fc.in_features)
# print("resnet18 output size: ", resnet18.fc.out_features)
