import torch
import os
from tqdm import tqdm


def save_checkpoint(model, epoch, config):
    os.makedirs(config['train']['output_dir'], exist_ok=True)
    path = os.path.join(config['train']['output_dir'],
                        f"checkpoint_{epoch}.pt")
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint: {path}")


def calculate_mean_std(dataset, batch_size=64, num_workers=2, pin_memory=True):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    mean = 0.
    std = 0.
    total = 0

    for images, _ in tqdm(loader, desc='Computing mean/std'):
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total += images.size(0)

    mean /= total
    std /= total
    return mean, std
