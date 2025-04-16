import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_mnist_dataloaders(batch_size=64, train=True, val_split=0.2):
    """
    Get MNIST dataloaders with optional augmentation and train/val split.

    Args:
        batch_size (int): Batch size for the dataloader
        train (bool): If True, load training data, else load test data
        val_split (float): Fraction of training data to use for validation

    Returns:
        tuple: (train_loader, val_loader) if train=True, else (test_loader,)
    """
    # Define transformations
    if train:
        transform = transforms.Compose([
            transforms.RandomRotation(10),  # Random rotation up to 10 degrees
            transforms.RandomAffine(degrees=0, translate=(
                0.1, 0.1)),  # Random translation
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    # Load dataset
    dataset = datasets.MNIST(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )

    if train:
        # Calculate split sizes
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        # Split dataset
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        return train_loader, val_loader
    else:
        # Create test dataloader
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        return test_loader,
