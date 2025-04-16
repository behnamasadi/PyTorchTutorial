import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.my_dataset import MyDataset
from models.my_model import MyModel
from trainers.trainer import Trainer
from utils.logger import setup_logging
from utils.metrics import get_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Deep Learning Project")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test", "predict"],
                        help="Mode to run the model in")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config):
    """Create necessary directories if they don't exist."""
    dirs = [
        config["data"]["train_path"],
        config["data"]["val_path"],
        config["data"]["test_path"],
        config["logging"]["log_dir"],
        config["logging"]["checkpoint_dir"],
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)

    # Setup directories
    setup_directories(config)

    # Setup device
    device = torch.device(config["device"])

    # Setup logging
    logger = setup_logging(config)

    # Create datasets and dataloaders
    train_dataset = MyDataset(config["data"]["train_path"])
    val_dataset = MyDataset(config["data"]["val_path"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"]
    )

    # Create model
    model = MyModel(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"]
    ).to(device)

    # Setup metrics
    metrics = get_metrics()

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        metrics=metrics,
        logger=logger
    )

    # Run training or evaluation
    if args.mode == "train":
        trainer.train()
    elif args.mode == "test":
        trainer.test()
    elif args.mode == "predict":
        trainer.predict()


if __name__ == "__main__":
    main()
