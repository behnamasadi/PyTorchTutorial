from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch

def calculate_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    mean = 0.
    std = 0.
    total_images = 0

    for images, _ in tqdm(loader, desc='Computing mean and std'):
        images = images.view(images.size(0), images.size(1), -1)  # (B, C, H*W)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += images.size(0)

    mean /= total_images
    std /= total_images
    return mean, std

if __name__ == '__main__':
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)

    mean, std = calculate_mean_std(dataset)
    print(f'Mean: {mean}')
    print(f'Std: {std}')

