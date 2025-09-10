import argparse
import pathlib
import yaml
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from utils.file_utils import resource_path, project_root


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed=seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return


def compute_mean_std_over_loader(data_loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    channel_sum = None
    channel_sq_sum = None
    num_pixels = 0

    with torch.no_grad():
        for images, _ in data_loader:
            # images: [B, C, H, W] on CPU (keep it there)
            images = images.to(dtype=torch.float64)  # accumulate in float64
            B, C, H, W = images.shape

            if channel_sum is None:
                channel_sum = torch.zeros(C, dtype=torch.float64)
                channel_sq_sum = torch.zeros(C, dtype=torch.float64)

            channel_sum += images.sum(dim=(0, 2, 3))
            channel_sq_sum += (images ** 2).sum(dim=(0, 2, 3))
            num_pixels += B * H * W

    mean = channel_sum / num_pixels
    var = (channel_sq_sum / num_pixels) - mean**2
    std = torch.sqrt(torch.clamp(var, min=0.0))

    return mean.float(), std.float()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute dataset mean/std on training split and persist to YAML")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for split and RNG")
    parser.add_argument("--data", type=str, required=False,
                        help="Path to dataset root for ImageFolder")
    parser.add_argument("--config", type=str, required=False, default=str(
        resource_path("config", "config_dev.yaml")), help="Path to YAML config to update")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Resize square side length")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size while computing stats")
    parser.add_argument("--train-split", type=float,
                        default=0.7, help="Fraction for training split")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction for validation split")
    args = parser.parse_args()

    # Resolve defaults using resource_path
    default_config = resource_path("config", "config_dev.yaml")
    default_data = resource_path(
        "data", "Brain_Cancer raw MRI data", "Brain_Cancer")

    print(default_config)
    print(default_data)

    print(pathlib.Path(default_config).exists())
    print(pathlib.Path(default_data).exists())

    seed = args.seed
    set_seed(seed=seed)
    generator = torch.Generator().manual_seed(seed)

    data_path = pathlib.Path(args.data) if args.data else default_data

    base_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    full_dataset = torchvision.datasets.ImageFolder(
        root=str(data_path), transform=base_transform)

    full_dataset_size = len(full_dataset)
    train_split = float(args.train_split)
    val_split = float(args.val_split)
    test_split = 1.0 - (train_split + val_split)

    training_size = int(train_split * full_dataset_size)
    val_size = int(val_split * full_dataset_size)
    test_size = full_dataset_size - (training_size + val_size)

    training_set, _, _ = random_split(
        dataset=full_dataset,
        lengths=[training_size, val_size, test_size],
        generator=generator,
    )

    data_loader = DataLoader(
        dataset=training_set, batch_size=args.batch_size, num_workers=0, pin_memory=False)

    mean, std = compute_mean_std_over_loader(data_loader)

    # Print results for immediate visibility
    print({"seed": seed, "mean": mean.tolist(), "std": std.tolist()})

    # Persist to YAML
    config_path = pathlib.Path(args.config) if args.config else default_config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("data_stats", {})
    # Store data_path relative to project root using pathlib for cross-platform compatibility
    relative_data_path = str(data_path.relative_to(project_root()))

    cfg["data_stats"].update({
        "seed": seed,
        "img_size": args.img_size,
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "data_path": relative_data_path,
    })

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Wrote data_stats to {config_path}")
