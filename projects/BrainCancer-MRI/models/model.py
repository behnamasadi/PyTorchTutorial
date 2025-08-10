import torch.nn as nn
import torchvision.models as models


def get_model(name, num_classes, weights):
    # weights is already processed in train.py, so use it directly

    if name == 'resnet18':
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        classifier_params = model.fc.parameters()

    elif name == 'vgg19_bn':
        model = models.vgg19_bn(weights=weights)
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, num_classes)
        classifier_params = model.classifier.parameters()

    elif name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes)
        classifier_params = model.classifier.parameters()

    else:
        raise ValueError(f"Unsupported model: {name}")

    return model, classifier_params
