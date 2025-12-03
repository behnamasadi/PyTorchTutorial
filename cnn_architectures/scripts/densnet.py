import timm
import torch.nn as nn

model = timm.create_model(
    'densenet201',
    pretrained=True,
    num_classes=5
)


in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, 5)
