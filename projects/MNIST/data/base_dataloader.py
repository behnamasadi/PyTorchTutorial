import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from abc import ABC, abstractmethod


class BaseMNISTDataLoader(ABC):
    def __init__(self, batch_size=64, num_workers=2):
        self.batch_size = batch_size
        self.num_workers = num_workers

    @abstractmethod
    def get_transforms(self, train=True):
        """Get the transforms for the dataset"""
        pass

    def get_mnist_dataloaders(self, train=True, val_split=0.2):
        """
        Get MNIST dataloaders with train/val split.

        Args:
            train (bool): If True, load training data, else load test data
            val_split (float): Fraction of training data to use for validation

        Returns:
            tuple: (train_loader, val_loader) if train=True, else (test_loader,)
        """
        # Get appropriate transforms
        transform = self.get_transforms(train)

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
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )

            return train_loader, val_loader
        else:
            # Create test dataloader
            test_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )

            return test_loader,
