# fmt: off
# isort: skip_file
# DO NOT reorganize imports - warnings filter must be FIRST!

import torch.nn.functional as F
import torch
import torch.nn as nn
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import timm
# fmt: on


def get_model(model_name, num_classes, pretrained=True):
    """
    Get model architecture with pre-trained weights and modify for classification.
    All models use RGB input (3 channels) for compatibility with pre-trained weights.

    Args:
        model_name (str): Model name
        num_classes (int): Number of output classes
        pretrained: Pre-trained weights

    Returns:
        model: The model with modified classifier
        classifier_params: Parameters of the classifier layer(s)
    """

    try:
        model = timm.create_model(model_name, pretrained=pretrained)

        # Get the number of features from the model (works for all timm models)
        num_features = model.num_features

        # num_features = 768 for all convnextv2 models
        # num_features = 1280 for all efficientnetv2 models
        # num_features = 440 for all RegNetY-4GF
        # num_features = 608 for all RegNetY-8GF

        # Create custom head for all models
        custom_head = nn.Sequential(
            # Global average pooling: [B, C, H, W] -> [B, C, 1, 1]
            nn.AdaptiveAvgPool2d(1),
            # [B, C, 1, 1] -> [B, C] (flatten from dim 1)
            nn.Flatten(1),
            nn.LayerNorm(num_features),
            nn.Dropout(0.3),
            # [B, num_features] -> [B, 128]
            nn.Linear(in_features=num_features, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.25),
            # [B, 128] -> [B, num_classes]
            nn.Linear(128, out_features=num_classes)
        )

        model.head = custom_head

    except RuntimeError as e:
        print(f"Warning: {e}")
        print(f"Unsupported model: {model_name}")

    return model
