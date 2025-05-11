# fmt: off
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import kagglehub
# fmt: on

import yaml
import os


# Read CSV file
# df = pd.read_csv(path + "/dataset.csv")
# print(df.head())


""" Function to calculate mean and std of a dataset"""


def calculate_mean_std(dataset, batch_size=64, num_workers=2, pin_memory=True):
    """Function to calculate mean and std of a dataset (channel-wise)"""

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    mean = 0.
    std = 0.
    total_images = 0

    for images, _ in tqdm(loader, desc='Computing mean and std'):
        # images shape is: B, C, H, W torch.Size([64, 3, 224, 224]

        images = images.view(images.size(0), images.size(1), -1)  # (B, C, H*W)

        mean += images.mean(2).sum(0)

        # since we want the mean over the image, we call images.mean(2), since
        # images.mean(0) -> B
        # images.mean(1) -> C

        # now images.mean(2) is the mean of each channel for all batches so it is something like:
        # [[0.0691, 0.0691, 0.0691],
        # [0.1690, 0.1690, 0.1690],
        # .
        # .
        # .
        # [0.1031, 0.1031, 0.1031],
        # [0.1088, 0.1088, 0.1088],

        # so  images.mean(2).sum(0) will collapse the 0 dimension which rows, so you get sum along rows:
        # images.mean(2).sum(0) -> tensor([7.3089, 7.3089, 7.3089])

        std += images.std(2).sum(0)
        total_images += images.size(0)

    mean /= total_images
    std /= total_images
    return mean, std


if __name__ == '__main__':

    # Get absolute path to this script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct path to config.yaml relative to this script
    config_path = os.path.join(current_dir, '..', 'config', 'config.yaml')

    # Safely normalize the path
    config_path = os.path.normpath(config_path)

    print(config_path)

    with open(config_path, 'r') as f:
        config_data = f.read()

    print(config_data)

    exit()

    # Download dataset
    path = kagglehub.dataset_download("orvile/brain-cancer-mri-dataset")
    print("Path to dataset files:", path)

    batch_size = 256  # or 512 or 1024
    num_workers = 4
    pin_memory = True

    image_size_h = 224
    image_size_w = 224

    # Basic transform for mean/std computation only
    initial_transform = transforms.Compose([
        transforms.Resize((image_size_h, image_size_w)),
        transforms.ToTensor()
    ])

    # Load full dataset with basic transform
    dataset_for_stats = datasets.ImageFolder(path, transform=initial_transform)

    # Split dataset
    train_size = int(0.7 * len(dataset_for_stats))
    val_size = int(0.15 * len(dataset_for_stats))
    test_size = len(dataset_for_stats) - train_size - val_size

    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        dataset_for_stats, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Compute mean and std using only training subset
    mean, std = calculate_mean_std(
        train_subset, batch_size=256, num_workers=4, pin_memory=True)

    print("Training Dataset mean:", mean.tolist())
    print("Training Dataset std:", std.tolist())
    print(f"Training set size: {len(train_subset)}")
    print(f"Validation set size: {len(val_subset)}")
    print(f"Test set size: {len(test_subset)}")

    # Define transforms using computed mean and std
    training_transform = transforms.Compose([
        transforms.Resize((image_size_h, image_size_w)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    validation_transform = transforms.Compose([
        transforms.Resize((image_size_h, image_size_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Reload dataset without transform so we can reassign them cleanly
    full_dataset = datasets.ImageFolder(path, transform=None)

    # Create new subsets with appropriate transforms
    train_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(path, transform=training_transform),
        indices=train_subset.indices
    )
    val_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(path, transform=validation_transform),
        indices=val_subset.indices
    )
    test_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(path, transform=validation_transform),
        indices=test_subset.indices
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Initialize VGG19 model
    model_vgg19_bn = models.vgg19_bn(
        weights=models.VGG19_BN_Weights.IMAGENET1K_V1)

    print("vgg19_bn input size: ", model_vgg19_bn.features.in_features)
    print("vgg19_bn output size: ", model_vgg19_bn.features.out_features)

    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    print("resnet18 input size: ", resnet18.fc.in_features)
    print("resnet18 output size: ", resnet18.fc.out_features)

    # # Freeze convolutional layers
    # for param in model_vgg19_bn.features.parameters():
    #     param.requires_grad = False
