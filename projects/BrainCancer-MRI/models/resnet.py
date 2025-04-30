from torchvision.models import resnet18
import torch.nn as nn


def get_resnet18(num_classes):
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
