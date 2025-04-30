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

# Set KaggleHub cache directory
# Ensure this environment variable is set before importing kagglehub
os.environ['KAGGLE_HUB_CACHE_DIR'] = os.path.abspath('../data/kagglehub/')

import kagglehub
# fmt: on


# Read CSV file
# df = pd.read_csv(path + "/dataset.csv")
# print(df.head())

# Function to calculate mean and std


def calculate_mean_std(dataset, batch_size=64, num_workers=2):
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

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

        # so  images.mean(2).sum(0) will collapse the 0 dimention which rows, so you get sum along rows:
        # images.mean(2).sum(0) -> tensor([7.3089, 7.3089, 7.3089])

        std += images.std(2).sum(0)
        total_images += images.size(0)

    mean /= total_images
    std /= total_images
    return mean, std


if __name__ == '__main__':
    print("KAGGLE_HUB_CACHE_DIR:", os.environ.get('KAGGLE_HUB_CACHE_DIR'))

    # Download latest version
    path = kagglehub.dataset_download("orvile/brain-cancer-mri-dataset")
    print("Path to dataset files:", path)

    # Set up paths and basic transform

    full_path = os.path.join(path, "Brain_Cancer raw MRI data", "Brain_Cancer")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        # Randomly rotate the image by 10 degrees
        transforms.RandomRotation(10),
        # Randomly change the brightness, contrast, saturation, and hue
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(full_path, transform=transform)

    # Load dataset

    mean, std = calculate_mean_std(dataset)
    print("Dataset mean:", mean)
    print("Dataset std:", std)

    # Update transform with calculated mean and std
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomVerticalFlip(),    # Randomly flip the image vertically
        # Randomly rotate the image by 10 degrees
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Update dataset with new transform
    dataset.transform = transform

    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4,
                             shuffle=False, num_workers=2)

    # Initialize VGG19 model
    model_vgg19_bn = models.vgg19_bn(
        weights=models.VGG19_BN_Weights.IMAGENET1K_V1)

    # Freeze convolutional layers
    for param in model_vgg19_bn.features.parameters():
        param.requires_grad = False

    print("Model architecture:")
    print(model_vgg19_bn)


# #model_vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)      # No batch norm
# #model_vgg19_bn = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1) # With batch norm

# #print(model_vgg19_bn.features[0])
# #print(model_vgg19_bn.parameters())

# # for param in model_vgg19_bn.classifier.parameters():
# #     print(param)

# #print(model_vgg19_bn.classifier)

# #exit()


# # Apply initial transform to resize and convert to tensor
# initial_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# MRI_dataset.transform = initial_transform

# # Calculate mean and std
# mean, std = get_mean_std(MRI_dataset)
# print(f"Mean: {mean}")
# print(f"Std: {std}")

# # Update transform with calculated mean and std
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std)
# ])

# MRI_dataset.transform = transform

# train_size = int(0.7 * len(MRI_dataset))
# val_size = int(0.15 * len(MRI_dataset))
# test_size = len(MRI_dataset) - train_size - val_size

# train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
#     MRI_dataset, [train_size, val_size, test_size]
# )

# print(f"Training set size: {len(train_dataset)}")
# print(f"Validation set size: {len(val_dataset)}")
# print(f"Test set size: {len(test_dataset)}")

# train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_data_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
# test_data_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# # class BrainCancerMRI(Dataset):
# #     def __init__(self, csv_file, root_dir, transform=None):
# #         self.df = pd.read_csv(csv_file)
# #         self.root_dir = root_dir
