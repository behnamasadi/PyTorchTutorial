#!/usr/bin/env python3
"""
Compute normalization statistics directly from the training split.

This script computes mean and standard deviation from training images
and writes them back to configs/data.yaml (normalization section).

For medical images, it handles grayscale → 3 channels conversion
to match pre-trained model expectations.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

# Allow running this file directly (python path/to/compute_normalization.py)
if __package__ in (None, ""):
    script_path = Path(__file__).resolve()
    package_root = script_path.parents[2]  # .../src
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import project utilities
from lung_disease_dataset.utils.paths import resource_path


def calculate_mean_std(
    dataset: datasets.ImageFolder,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate mean and standard deviation across all channels of the dataset.

    Args:
        dataset: ImageFolder dataset
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory in DataLoader

    Returns:
        Tuple of (mean tensor, std tensor) of shape [num_channels]
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    print("🔄 Computing statistics...")
    for batch_idx, (images, _) in enumerate(dataloader):
        # images shape: [batch_size, channels, height, width]
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)

        # Compute mean per channel
        mean += images.mean(2).sum(0)
        total_samples += batch_samples

        if (batch_idx + 1) % 10 == 0:
            print(f"   Processed {(batch_idx + 1) * batch_size} images...")

    mean /= total_samples

    # Compute std
    for batch_idx, (images, _) in enumerate(dataloader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        std += ((images - mean.unsqueeze(1)) ** 2).mean(2).sum(0)

    std = torch.sqrt(std / total_samples)

    return mean, std


def build_transform(img_size: int, grayscale_to_rgb: bool = True) -> transforms.Compose:
    """
    Build transform pipeline matching the training preprocessing.

    For medical images (grayscale), converts to 3-channel RGB by repeating
    the grayscale channel 3 times (standard practice for pre-trained models).

    Args:
        img_size: Target image size
        grayscale_to_rgb: If True, convert grayscale to 3-channel RGB

    Returns:
        Transform pipeline
    """
    tfms = [
        transforms.Resize((img_size, img_size)),
    ]

    if grayscale_to_rgb:
        # Convert grayscale to 3-channel RGB by repeating channels
        # This matches the medical imaging best practice
        tfms.append(transforms.Grayscale(num_output_channels=3))
    else:
        tfms.append(transforms.Lambda(lambda img: img.convert("RGB")))

    tfms.append(transforms.ToTensor())

    return transforms.Compose(tfms)


def load_training_dataset(
    dataset_dir: Path,
    img_size: int,
    grayscale_to_rgb: bool = True,
) -> datasets.ImageFolder:
    """
    Create a torchvision ImageFolder for the training directory.

    Args:
        dataset_dir: Path to training dataset
        img_size: Target image size
        grayscale_to_rgb: Convert grayscale to 3-channel RGB

    Returns:
        ImageFolder dataset
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if not any(dataset_dir.iterdir()):
        raise ValueError(f"Dataset directory is empty: {dataset_dir}")

    transform = build_transform(img_size, grayscale_to_rgb)
    return datasets.ImageFolder(str(dataset_dir), transform=transform)


def compute_stats_from_train(
    dataset_dir: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    img_size: int,
    grayscale_to_rgb: bool = True,
) -> Tuple[datasets.ImageFolder, torch.Tensor, torch.Tensor]:
    """
    Compute mean/std tensors from the training directory.

    Args:
        dataset_dir: Path to training dataset
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        img_size: Target image size
        grayscale_to_rgb: Convert grayscale to 3-channel RGB

    Returns:
        Tuple of (dataset, mean tensor, std tensor)
    """
    dataset = load_training_dataset(dataset_dir, img_size, grayscale_to_rgb)

    print(f"📂 Dataset path: {dataset_dir}")
    print(f"🧾 Found classes: {dataset.classes}")
    print(f"📸 Total training images: {len(dataset)}")
    print(f"🎨 Image size: {img_size}x{img_size}")
    print(f"🔢 Channels: 3 (RGB)" +
          (" [grayscale → RGB]" if grayscale_to_rgb else ""))

    mean, std = calculate_mean_std(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataset, mean, std


def update_data_config(mean_list: list, std_list: list, data_config_path: Path):
    """
    Update data.yaml with computed normalization constants.

    Args:
        mean_list: List of mean values per channel
        std_list: List of std values per channel
        data_config_path: Path to data.yaml
    """
    if not data_config_path.exists():
        print(
            f"⚠️  data.yaml not found at {data_config_path}, skipping config update")
        return

    with open(data_config_path) as f:
        config = yaml.safe_load(f)

    # Update normalization constants
    if "normalization" not in config:
        config["normalization"] = {}

    config["normalization"]["mean"] = mean_list
    config["normalization"]["std"] = std_list

    # Add comment
    config["normalization"]["_comment"] = "Auto-computed by compute_normalization.py"

    with open(data_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False,
                  sort_keys=False, indent=2)

    print(
        f"✅ Updated {data_config_path} with computed normalization constants")


def load_config(config_path: Optional[Path] = None) -> dict:
    """
    Load configuration from data.yaml.

    Args:
        config_path: Path to data.yaml (if None, uses default)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = resource_path("configs", "data.yaml")

    if not config_path.exists():
        print(f"⚠️  Config file not found at {config_path}, using defaults")
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute normalization stats from training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults from data.yaml
  python -m lung_disease_dataset.utils.compute_normalization
  
  # Specify custom dataset path
  python -m lung_disease_dataset.utils.compute_normalization --dataset ./data/train
  
  # Use ImageNet defaults (skip computation)
  python -m lung_disease_dataset.utils.compute_normalization --use-imagenet
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to data.yaml config file (default: configs/data.yaml)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to training dataset directory (overrides config)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Image size (default: from data.yaml or 224)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for DataLoader (default: from data.yaml or 64)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers (default: from data.yaml or 4)",
    )
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        help="Enable pinned memory",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pinned memory",
    )
    parser.add_argument(
        "--no-grayscale-to-rgb",
        dest="grayscale_to_rgb",
        action="store_false",
        help="Disable grayscale to RGB conversion (default: enabled for medical images)",
    )
    parser.add_argument(
        "--no-update-config",
        action="store_true",
        help="Skip updating data.yaml with computed values",
    )
    parser.add_argument(
        "--use-imagenet",
        action="store_true",
        help="Use ImageNet normalization constants (skip computation)",
    )

    parser.set_defaults(pin_memory=True, grayscale_to_rgb=True)
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config) if args.config else resource_path(
        "configs", "data.yaml")
    config = load_config(config_path)

    # Use ImageNet defaults if requested
    if args.use_imagenet:
        mean_list = [0.485, 0.456, 0.406]
        std_list = [0.229, 0.224, 0.225]
        print("📊 Using ImageNet normalization constants:")
        print(f"   Mean: {mean_list}")
        print(f"   Std:  {std_list}")

        if not args.no_update_config:
            update_data_config(mean_list, std_list, config_path)

        return

    # Get dataset path
    if args.dataset:
        dataset_path = Path(args.dataset).resolve()
    else:
        dataset_path = resource_path(config.get(
            "paths", {}).get("train", "data/train"))

    # Get parameters from config or args
    img_size = args.img_size or config.get(
        "dataset", {}).get("image_size", 224)
    batch_size = args.batch_size or config.get(
        "dataloader", {}).get("batch_size", 64)
    num_workers = args.num_workers or config.get(
        "dataloader", {}).get("num_workers", 4)
    pin_memory = args.pin_memory if args.pin_memory is not None else config.get(
        "dataloader", {}).get("pin_memory", True)

    # Determine grayscale_to_rgb from config
    if args.grayscale_to_rgb is None:
        grayscale_to_rgb = config.get("dataset", {}).get(
            "normalize_grayscale", True)
    else:
        grayscale_to_rgb = args.grayscale_to_rgb

    try:
        dataset, mean, std = compute_stats_from_train(
            dataset_dir=dataset_path,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            img_size=img_size,
            grayscale_to_rgb=grayscale_to_rgb,
        )
    except Exception as e:
        print(f"❌ Failed to compute statistics: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    mean_list = mean.tolist()
    std_list = std.tolist()

    print("\n✅ Statistics computed successfully!")
    print(f"📊 Mean: {mean_list}")
    print(f"📉 Std:  {std_list}")

    if not args.no_update_config:
        update_data_config(mean_list, std_list, config_path)
    else:
        print("⚠️  Config update disabled via --no-update-config; not writing data.yaml.")


if __name__ == "__main__":
    main()
