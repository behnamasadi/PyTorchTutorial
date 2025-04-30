from torchvision.models import vgg19
import torch.nn as nn


def get_vgg19(num_classes):
    model = vgg19(pretrained=True)
    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features, num_classes)
    return model
