import torchvision.models as models
import torch
import torch.nn as nn
from typing import Optional


def get_model(model_name: str, num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """
    Get a model by name with proper configuration.
    
    Args:
        model_name: Name of the model to load
        num_classes: Number of output classes (default: 3 for brain MRI)
        pretrained: Whether to use pretrained weights
        
    Returns:
        Configured model ready for training
        
    Raises:
        ValueError: If model_name is not supported
    """
    model_name = model_name.lower()
    
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        # Modify the classifier for our number of classes
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        # Modify the classifier for our number of classes
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        
    else:
        supported_models = ["mobilenet_v2", "efficientnet_b0"]
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {supported_models}")
    
    return model
# mobilenet_v2 = models.mobilenet_v2(weights=torch.MobileNet_V2_Weights.DEFAULT)
# mobilenetv2.parameters
