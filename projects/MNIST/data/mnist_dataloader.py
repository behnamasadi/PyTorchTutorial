import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(model_type='mlp'):
    """
    Get the appropriate transforms based on the model type.

    Args:
        model_type (str): Type of model to use transforms for. Options: 'mlp', 'resnet'

    Returns:
        transforms.Compose: The composed transforms
    """
    if model_type == 'mlp':
        return transforms.Compose([
            transforms.RandomRotation(10),  # Random rotation up to 10 degrees
            transforms.RandomAffine(degrees=0, translate=(
                0.1, 0.1)),  # Random translation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
    elif model_type == 'resnet':
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match ResNet input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_mnist_dataloaders(batch_size=32, num_workers=4, model_type='mlp'):
    """
    Creates train and test dataloaders for MNIST dataset.

    Args:
        batch_size (int): Number of samples per batch. Default is 32.
        num_workers (int): Number of subprocesses to use for data loading. Default is 4.
        model_type (str): Type of model to use appropriate transforms for. Options: 'mlp', 'resnet'

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Get the appropriate transforms
    transform = get_transforms(model_type)

    # Download and load the training data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Download and load the test data
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
