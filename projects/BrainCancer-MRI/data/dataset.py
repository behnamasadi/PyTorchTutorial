import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, Subset


def get_transforms(mean, std, img_size, augment):
    train_tf = [transforms.Resize((img_size, img_size))]
    if augment:
        train_tf += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        ]
    train_tf += [transforms.ToTensor()]
    if mean is not None and std is not None:
        train_tf += [transforms.Normalize(mean, std)]
    return transforms.Compose(train_tf), transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def load_datasets(config, mean=None, std=None):
    path = config['dataset']['path']
    img_size = config['dataset']['img_size']
    augment = config['transform']['augmentation']

    transform_train, transform_val_test = get_transforms(
        mean, std, img_size, augment)

    full_dataset = datasets.ImageFolder(path, transform=None)
    n_total = len(full_dataset)
    n_train = int(n_total * config['dataset']['train_split'])
    n_val = int(n_total * config['dataset']['val_split'])
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(config['dataset']['seed'])
    train_split, val_split, test_split = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator)

    train_ds = Subset(datasets.ImageFolder(
        path, transform=transform_train), indices=train_split.indices)
    val_ds = Subset(datasets.ImageFolder(
        path, transform=transform_val_test), indices=val_split.indices)
    test_ds = Subset(datasets.ImageFolder(
        path, transform=transform_val_test), indices=test_split.indices)

    return train_ds, val_ds, test_ds
