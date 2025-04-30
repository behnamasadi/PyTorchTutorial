from models.vgg19 import get_vgg19
from models.resnet import get_resnet18


def get_model(name: str, num_classes: int):
    name = name.lower()
    if name == 'vgg19':
        return get_vgg19(num_classes)
    elif name == 'resnet18':
        return get_resnet18(num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")
