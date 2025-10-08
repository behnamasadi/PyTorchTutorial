"""Training script for TGS Salt segmentation using U-Net with optimized loss weighting.

This script follows ML workflow best practices including:
- Comprehensive type annotations
- Google-style docstrings
- Proper error handling and logging
- Modular design with single responsibility functions
- Experiment tracking with Weights & Biases
- Configuration management
"""

from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict, Any
import logging
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tgs_salt_dataset import TgsSaltSemanticDataset
from models import unet
from utils.file_utils import resource_path
from loss_functions.soft_dice import SoftDiceLoss
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"


def setup_training_config() -> Dict[str, Any]:
    """Set up training configuration parameters.

    Returns:
        Dict containing all training hyperparameters and settings.
    """
    return {
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "epochs": 50,
        "train_split": 0.85,
        "val_split": 0.15,
        "pad_to": 32,
        "loss_weights": {"ce": 0.3, "dice": 2.0},
        "scheduler_config": {
            "factor": 0.7,
            "patience": 10,
            "min_lr": 1e-6
        },
        "gradient_clip_norm": 1.0,
        "checkpoint_frequency": 10,
        "early_stopping_patience": 15,
        "best_model_path": "best_model.pth"
    }


def create_data_loaders(
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders.

    Args:
        config: Training configuration dictionary.

    Returns:
        Tuple of (train_dataloader, validation_dataloader).

    Raises:
        RuntimeError: If dataset cannot be loaded or split.
    """
    try:
        project_root = Path(resource_path(""))
        train_data_path = "data/tgs_salt/trainSuper"
        root = project_root.joinpath(train_data_path)

        # Image transforms
        img_transform = transforms.Compose([transforms.ToTensor()])

        # Create dataset
        dataset = TgsSaltSemanticDataset(
            root, "images", "masks",
            transform=img_transform,
            pad_to=config["pad_to"]
        )

        # Split dataset
        dataset_len = len(dataset)
        train_size = int(config["train_split"] * dataset_len)
        validation_size = dataset_len - train_size

        train_dataset, validation_dataset = random_split(
            dataset,
            lengths=[train_size, validation_size],
            generator=torch.Generator()
        )

        # Create data loaders
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config["batch_size"],
            pin_memory=True,
            num_workers=4
        )

        validation_dataloader = DataLoader(
            dataset=validation_dataset,
            batch_size=config["batch_size"],
            pin_memory=True,
            num_workers=4
        )

        logger.info(
            f"Created data loaders: {len(train_dataset)} train, {len(validation_dataset)} validation samples")
        return train_dataloader, validation_dataloader

    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        raise RuntimeError(f"Data loader creation failed: {e}") from e


def train_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_criterion: nn.Module,
    dice_criterion: nn.Module,
    config: Dict[str, Any],
    epoch: int
) -> Tuple[float, float, float]:
    """Train model for one epoch.

    Args:
        model: The neural network model.
        train_dataloader: Training data loader.
        optimizer: Optimizer for model parameters.
        ce_criterion: Cross-entropy loss function.
        dice_criterion: Dice loss function.
        config: Training configuration.
        epoch: Current epoch number.

    Returns:
        Tuple of (average_total_loss, average_ce_loss, average_dice_loss).
    """
    model.train()
    epoch_total_loss = 0.0
    epoch_ce_loss = 0.0
    epoch_dice_loss = 0.0

    for batch_idx, (images, targets) in enumerate(train_dataloader):
        try:
            images = images.to(device)
            targets = targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(images)

            # Compute losses
            ce_loss = ce_criterion(logits, targets)
            dice_loss = dice_criterion(logits, targets)
            total_loss = (config["loss_weights"]["ce"] * ce_loss +
                          config["loss_weights"]["dice"] * dice_loss)

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["gradient_clip_norm"]
            )

            # Update weights
            optimizer.step()

            # Accumulate losses
            epoch_total_loss += total_loss.item()
            epoch_ce_loss += ce_loss.item()
            epoch_dice_loss += dice_loss.item()

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                dice_score = 1 - dice_loss.item()  # Convert loss to score
                print(f'Epoch [{epoch+1}/{config["epochs"]}], Batch [{batch_idx}/{len(train_dataloader)}], '
                      f'Loss: {total_loss.item():.4f}, CE: {ce_loss.item():.4f}, Dice Loss: {dice_loss.item():.4f}, Dice Score: {dice_score:.4f}')

        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            raise

    # Calculate averages
    avg_total_loss = epoch_total_loss / len(train_dataloader)
    avg_ce_loss = epoch_ce_loss / len(train_dataloader)
    avg_dice_loss = epoch_dice_loss / len(train_dataloader)

    return avg_total_loss, avg_ce_loss, avg_dice_loss


def validate_epoch(
    model: nn.Module,
    validation_dataloader: DataLoader,
    ce_criterion: nn.Module,
    dice_criterion: nn.Module,
    config: Dict[str, Any]
) -> Tuple[float, float, float]:
    """Validate model for one epoch.

    Args:
        model: The neural network model.
        validation_dataloader: Validation data loader.
        ce_criterion: Cross-entropy loss function.
        dice_criterion: Dice loss function.
        config: Training configuration.

    Returns:
        Tuple of (validation_total_loss, validation_ce_loss, validation_dice_loss).
    """
    model.eval()
    val_total_loss = 0.0
    val_ce_loss = 0.0
    val_dice_loss = 0.0

    with torch.no_grad():
        for images, targets in validation_dataloader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            ce_loss = ce_criterion(logits, targets)
            dice_loss = dice_criterion(logits, targets)
            total_loss = (config["loss_weights"]["ce"] * ce_loss +
                          config["loss_weights"]["dice"] * dice_loss)

            val_total_loss += total_loss.item()
            val_ce_loss += ce_loss.item()
            val_dice_loss += dice_loss.item()

    # Calculate averages
    avg_val_total_loss = val_total_loss / len(validation_dataloader)
    avg_val_ce_loss = val_ce_loss / len(validation_dataloader)
    avg_val_dice_loss = val_dice_loss / len(validation_dataloader)

    return avg_val_total_loss, avg_val_ce_loss, avg_val_dice_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_path: str
) -> None:
    """Save model checkpoint.

    Args:
        model: The neural network model.
        optimizer: Optimizer state.
        epoch: Current epoch number.
        loss: Current loss value.
        checkpoint_path: Path to save checkpoint.
    """
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        logger.info(f'Model saved to {checkpoint_path}')
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        raise


def save_best_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    best_val_loss: float,
    config: Dict[str, Any]
) -> float:
    """Save best model if current validation loss is better.

    Args:
        model: The neural network model.
        optimizer: Optimizer state.
        epoch: Current epoch number.
        val_loss: Current validation loss.
        best_val_loss: Best validation loss so far.
        config: Training configuration.

    Returns:
        Updated best validation loss.
    """
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            config["best_model_path"]
        )
        logger.info(f'New best model saved! Validation loss: {val_loss:.4f}')

    return best_val_loss


def should_early_stop(
    val_loss: float,
    best_val_loss: float,
    patience_counter: int,
    patience: int
) -> Tuple[bool, int]:
    """Check if training should stop early.

    Args:
        val_loss: Current validation loss.
        best_val_loss: Best validation loss so far.
        patience_counter: Current patience counter.
        patience: Maximum patience before stopping.

    Returns:
        Tuple of (should_stop, updated_patience_counter).
    """
    if val_loss < best_val_loss:
        patience_counter = 0
    else:
        patience_counter += 1

    should_stop = patience_counter >= patience
    return should_stop, patience_counter


def main() -> None:
    """Main training function following ML workflow best practices."""
    try:
        # Setup configuration
        config = setup_training_config()
        logger.info(f"Training configuration: {config}")

        # Create data loaders
        train_dataloader, validation_dataloader = create_data_loaders(config)

        # Initialize model
        model = unet.UNet(input_channel=3, n_classes=2, base=64)
        model = model.to(device)
        logger.info(f"Model initialized on device: {device}")

        # Setup optimizer and scheduler
        optimizer = AdamW(
            params=model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config["scheduler_config"]["factor"],
            patience=config["scheduler_config"]["patience"],
            verbose=True,
            min_lr=config["scheduler_config"]["min_lr"]
        )

        # Initialize loss functions
        ce_criterion = nn.CrossEntropyLoss(ignore_index=255)
        dice_criterion = SoftDiceLoss()

        # Initialize Weights & Biases
        wandb.init(
            project="tgs-salt-segmentation",
            config={
                **config,
                "model": "UNet",
                "dataset": "TGS Salt",
                "device": device
            }
        )
        logger.info("Weights & Biases initialized")

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(config["epochs"]):
            try:
                # Train epoch
                train_total_loss, train_ce_loss, train_dice_loss = train_epoch(
                    model, train_dataloader, optimizer,
                    ce_criterion, dice_criterion, config, epoch
                )

                # Validate epoch
                val_total_loss, val_ce_loss, val_dice_loss = validate_epoch(
                    model, validation_dataloader,
                    ce_criterion, dice_criterion, config
                )

                # Calculate dice score
                dice_score = 1 - val_dice_loss

                # Log epoch metrics (detailed format like old training)
                logger.info(f'Epoch [{epoch+1}/{config["epochs"]}] - '
                            f'Loss: {train_total_loss:.4f}, '
                            f'CE: {train_ce_loss:.4f}, '
                            f'Dice Loss: {train_dice_loss:.4f}, '
                            f'Dice Score: {dice_score:.4f}')

                # Also print validation metrics
                logger.info(f'Validation - Loss: {val_total_loss:.4f}, '
                            f'CE: {val_ce_loss:.4f}, '
                            f'Dice Loss: {val_dice_loss:.4f}')

                # Log to wandb (clean logging without batch metrics)
                wandb.log({
                    "loss": train_total_loss,
                    "epoch_ce_loss": train_ce_loss,
                    "epoch_dice_loss": train_dice_loss,
                    "val_loss": val_total_loss,
                    "val_ce_loss": val_ce_loss,
                    "val_dice_loss": val_dice_loss,
                    "dice_score": dice_score,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

                # Update learning rate scheduler
                scheduler.step(val_total_loss)

                # Save best model
                best_val_loss = save_best_model(
                    model, optimizer, epoch + 1, val_total_loss,
                    best_val_loss, config
                )

                # Save regular checkpoint
                if (epoch + 1) % config["checkpoint_frequency"] == 0:
                    checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch+1}.pth'
                    save_checkpoint(model, optimizer, epoch + 1,
                                    train_total_loss, checkpoint_path)

                # Check for early stopping
                should_stop, patience_counter = should_early_stop(
                    val_total_loss, best_val_loss, patience_counter,
                    config["early_stopping_patience"]
                )

                if should_stop:
                    logger.info(f'Early stopping triggered after {epoch + 1} epochs. '
                                f'No improvement for {config["early_stopping_patience"]} epochs.')
                    break

                logger.info('-' * 50)

            except Exception as e:
                logger.error(f"Error in epoch {epoch + 1}: {e}")
                raise

        logger.info(
            f'Training completed! Best model saved to {config["best_model_path"]}')

        # Finish wandb run
        wandb.finish()

    except Exception as e:
        logger.error(f"Training failed: {e}")
        wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
