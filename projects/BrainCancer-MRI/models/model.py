import torch.nn as nn
import torchvision.models as models
import torch





def get_model(name, num_classes, weights):
    """
    Get model architecture with pre-trained weights and modify for classification.
    All models use RGB input (3 channels) for compatibility with pre-trained weights.

    Args:
        name (str): Model name
        num_classes (int): Number of output classes
        weights: Pre-trained weights

    Returns:
        model: The model with modified classifier
        classifier_params: Parameters of the classifier layer(s)
    """

    if name == 'resnet18':
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        classifier_params = model.fc.parameters()

    elif name == 'resnet50':
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        classifier_params = model.fc.parameters()

    elif name == 'swin_t':
        model = models.swin_t(weights=weights)
        # Note: Swin Transformer doesn't have a simple conv1 layer to modify
        # You might need to handle this differently or keep RGB conversion
        model.head = nn.Linear(model.head.in_features, num_classes)
        # Initialize classifier head with smaller weights for stability
        nn.init.normal_(model.head.weight, mean=0.0, std=0.01)
        nn.init.constant_(model.head.bias, 0.0)
        classifier_params = model.head.parameters()

    elif name == 'swin_s':
        model = models.swin_s(weights=weights)
        # Note: Swin Transformer doesn't have a simple conv1 layer to modify
        model.head = nn.Linear(model.head.in_features, num_classes)
        classifier_params = model.head.parameters()

    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes)
        classifier_params = model.classifier.parameters()

    elif name == 'vit_b_16':
        model = models.vit_b_16(weights=weights)
        # Note: Vision Transformer doesn't have a simple conv1 layer to modify
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        classifier_params = model.heads.parameters()

    elif name == 'medical_cnn':
        # Custom CNN inspired by Xception for medical images
        from torchvision.models import efficientnet_b0
        base_model = efficientnet_b0(weights=weights)

        # Get feature dimension from the backbone
        feature_dim = base_model.classifier[1].in_features  # Should be 1280

        # Replace classifier with medical-optimized one
        base_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

        model = base_model

        # Freeze backbone, only train classifier
        for param in model.features.parameters():
            param.requires_grad = False

        # Only classifier parameters are trainable
        classifier_params = model.classifier.parameters()

        # Initialize classifier properly
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.constant_(layer.bias, 0.0)

    elif name == 'vgg19_bn':
        model = models.vgg19_bn(weights=weights)
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, num_classes)
        classifier_params = model.classifier.parameters()

    elif name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=weights)
        # MobileNet has a different structure, would need specific handling
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes)
        classifier_params = model.classifier.parameters()

    elif name == 'xception_medical':
        # Xception-inspired model for medical images (299x299 input)
        # PyTorch doesn't have Xception, so using EfficientNet-B0 as similar architecture
        from torchvision.models import efficientnet_b0

        # Load pretrained backbone
        base_model = efficientnet_b0(weights=weights)



        # Get feature dimension after adaptive pooling
        # 1280 for EfficientNet-B0
        feature_dim = base_model.classifier[1].in_features

        # Create Kaggle-inspired classifier (matches the TensorFlow Sequential model)
        # TF model: [base_model, Flatten(), Dropout(0.3), Dense(128, relu), Dropout(0.25), Dense(4, softmax)]
        classifier = nn.Sequential(
            # Global max pooling (like TF pooling='max')
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
            # Note: No softmax here - CrossEntropyLoss includes it
        )

        # Replace the original classifier
        base_model.classifier = classifier
        model = base_model

        # Keep backbone trainable (like in Kaggle - layers not frozen)
        # This matches the commented out freezing in the original code
        classifier_params = model.parameters()  # Train all parameters

        # Initialize classifier layers properly
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.constant_(layer.bias, 0.0)

    else:
        raise ValueError(f"Unsupported model: {name}. Supported models: "
                         f"resnet18, resnet50, swin_t, swin_s, efficientnet_b0, "
                         f"vit_b_16, vgg19_bn, mobilenet_v2, medical_cnn, xception_medical")

    return model, classifier_params


def get_model_info(name):
    """
    Get information about model architecture.

    Args:
        name (str): Model name

    Returns:
        dict: Model information
    """
    model_info = {
        'resnet18': {
            'type': 'CNN',
            'params': '11.7M',
            'description': 'Residual Network with 18 layers'
        },
        'resnet50': {
            'type': 'CNN',
            'params': '25.6M',
            'description': 'Residual Network with 50 layers'
        },
        'swin_t': {
            'type': 'Transformer',
            'params': '28.3M',
            'description': 'Swin Transformer Tiny'
        },
        'swin_s': {
            'type': 'Transformer',
            'params': '49.6M',
            'description': 'Swin Transformer Small'
        },
        'efficientnet_b0': {
            'type': 'CNN',
            'params': '5.3M',
            'description': 'EfficientNet B0 - efficient CNN'
        },
        'vit_b_16': {
            'type': 'Transformer',
            'params': '86.6M',
            'description': 'Vision Transformer Base with 16x16 patches'
        },
        'vgg19_bn': {
            'type': 'CNN',
            'params': '143.7M',
            'description': 'VGG19 with Batch Normalization'
        },
        'mobilenet_v2': {
            'type': 'CNN',
            'params': '3.5M',
            'description': 'MobileNet V2 - lightweight CNN'
        },
        'medical_cnn': {
            'type': 'Medical CNN',
            'params': '5.3M (frozen) + 0.2M (trainable)',
            'description': 'EfficientNet backbone + medical-optimized classifier'
        },
        'xception_medical': {
            'type': 'Xception-inspired Medical CNN',
            'params': '5.3M (fully trainable)',
            'description': 'Kaggle solution: EfficientNet backbone + Xception-style classifier (299x299)',
            'optimizer': 'Adamax',
            'input_size': '299x299'
        }
    }

    return model_info.get(name, {'type': 'Unknown', 'params': 'Unknown', 'description': 'Unknown model'})
