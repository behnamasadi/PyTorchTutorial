import torchvision.models as models
import torch
import torch.nn as nn
from torchvision.models import WeightsEnum


class xception_medical(nn.Module):
    def __init__(self, num_classes: int, weights: WeightsEnum):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=weights)

        feature_dim = self.backbone.classifier[1].in_features  # Should be 1280

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features=feature_dim, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

        self.backbone.classifier = self.classifier

        # Initialize classifier layers properly, does this model non deterministic
        for layer in self.backbone.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.constant_(layer.bias, 0.0)

        return

    def forward(self, x):
        return self.backbone(x)


def get_model(model_name: str, weights: WeightsEnum, num_classes: int) -> torch.nn.Module:
    if model_name == "efficientnet_b0":
        return models.efficientnet_b0(weights=weights)
    elif model_name == "mobilenet_v2":
        return models.mobilenet_v2(weights=weights)
    elif model_name == "xception_medical":
        return xception_medical(num_classes=num_classes, weights=weights)
    else:
        raise ValueError(f"Unknown model: {model_name}")
