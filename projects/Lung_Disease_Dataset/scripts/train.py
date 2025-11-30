#!/usr/bin/env python3
"""
Minimal two-stage training script for the Lungs Disease Dataset.
"""

import yaml
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import Adam, AdamW
import torch.nn as nn
import torch
import argparse
import sys
import time
import warnings
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional
import shutil

# Optional kagglehub import
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    kagglehub = None

# Suppress pydantic warnings from wandb/mlflow dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


# Optional logging imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_runpod.yaml"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lung_disease_dataset.models.model import get_model  # noqa: E402


@dataclass
class StageConfig:
    name: str
    enabled: bool
    freeze_backbone: bool
    epochs: int
    learning_rate: float
    lr_schedule: str
    lr_schedule_params: Dict[str, float]
    head_lr: float = None  # Optional: separate LR for head
    backbone_lr: float = None  # Optional: separate LR for backbone
    early_stop_patience: int = None  # Optional: early stopping patience


def load_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path_str.lstrip("./")
    return path.resolve()


def download_dataset_if_needed(train_path: Path, val_path: Path) -> None:
    """
    Download the dataset from Kaggle if it doesn't already exist.
    Checks if train and val directories exist and have content before downloading.

    Uses KAGGLE_USERNAME and KAGGLE_KEY environment variables for authentication.
    """
    # Check if data already exists
    train_exists = train_path.exists() and any(train_path.iterdir())
    val_exists = val_path.exists() and any(val_path.iterdir())

    if train_exists and val_exists:
        print(f"✅ Dataset already exists at:")
        print(f"   Train: {train_path}")
        print(f"   Val: {val_path}")
        return

    if not KAGGLEHUB_AVAILABLE:
        raise ImportError(
            "kagglehub is required for automatic dataset download. "
            "Install it with: pip install kagglehub"
        )

    # Verify Kaggle credentials are available
    # Works both locally (kaggle.json) and on RunPod (env vars)
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    if kaggle_username and kaggle_key:
        print(
            f"✅ Kaggle credentials found in environment (username: {kaggle_username[:3]}...)")
        print("   Using KAGGLE_USERNAME and KAGGLE_KEY for authentication")
    else:
        print("ℹ️  KAGGLE_USERNAME/KAGGLE_KEY not in environment.")
        print("   Attempting to use kaggle.json file or other authentication methods...")
        print("   (This is normal for local development)")

    print("Downloading dataset from Kaggle...")
    try:
        # Download the dataset (kagglehub handles caching automatically)
        # kagglehub will use env vars if available, otherwise falls back to kaggle.json
        dataset_path = kagglehub.dataset_download(
            "khaleddev/lungs-disease-dataset-broken")
        print(f"✅ Dataset downloaded to: {dataset_path}")

        # The dataset might be in a zip file or already extracted
        # Check if we need to extract or organize it
        dataset_path = Path(dataset_path)

        # Look for train/val/test directories in the downloaded dataset
        # The structure might vary, so we'll search for common patterns
        possible_train_dirs = list(dataset_path.rglob("train"))
        possible_val_dirs = list(dataset_path.rglob("val"))
        possible_test_dirs = list(dataset_path.rglob("test"))

        # If train/val directories are found in the dataset, copy them
        if possible_train_dirs and possible_val_dirs:
            train_src = possible_train_dirs[0]
            val_src = possible_val_dirs[0]

            # Create parent directories if needed
            train_path.parent.mkdir(parents=True, exist_ok=True)
            val_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy train data
            if not train_path.exists():
                print(f"Copying train data from {train_src} to {train_path}")
                shutil.copytree(train_src, train_path)
            else:
                print("Train directory already exists, skipping copy")

            # Copy val data
            if not val_path.exists():
                print(
                    f"Copying validation data from {val_src} to {val_path}")
                shutil.copytree(val_src, val_path)
            else:
                print("Validation directory already exists, skipping copy")

            # Handle test directory if it exists
            if possible_test_dirs:
                test_path = val_path.parent / "test"
                test_src = possible_test_dirs[0]
                if not test_path.exists():
                    print(
                        f"Copying test data from {test_src} to {test_path}")
                    shutil.copytree(test_src, test_path)
        else:
            # If the dataset structure is different, check for class folders at root
            # and organize them into train/val/test if needed
            class_folders = [d for d in dataset_path.iterdir()
                             if d.is_dir() and not d.name.startswith('.')]

            if class_folders:
                print(
                    f"❌ Dataset structure differs from expected. Found {len(class_folders)} folders.")
                print(
                    f"   Please organize the data manually or update the download function.")
                print(f"   Dataset location: {dataset_path}")
            else:
                print("❌ Could not find train/val directories in downloaded dataset.")
                print(f"   Dataset location: {dataset_path}")
                print(f"   Please check the dataset structure and organize manually.")

        print("✅ Dataset preparation complete")

    except Exception as e:
        error_msg = (
            f"Failed to download or organize dataset: {e}\n"
            f"\nAuthentication options:\n"
            f"  - RunPod/Docker: Set KAGGLE_USERNAME and KAGGLE_KEY environment variables\n"
            f"  - Local: Use kagglehub.login() or create ~/.kaggle/kaggle.json\n"
            f"  - Or ensure dataset is already downloaded to: {train_path.parent}"
        )
        raise RuntimeError(error_msg) from e


def prepare_stage(name: str, stage_cfg: Dict, *, freeze: bool, epochs: int, lr: float) -> StageConfig:
    cfg = stage_cfg or {}
    # Ensure learning_rate is always a float (YAML may parse scientific notation as string)
    learning_rate = cfg.get("learning_rate", lr)
    learning_rate = float(learning_rate) if learning_rate is not None else lr

    # Handle separate learning rates for head and backbone
    head_lr = cfg.get("head_lr")
    if head_lr is not None:
        head_lr = float(head_lr)

    backbone_lr = cfg.get("backbone_lr")
    if backbone_lr is not None:
        backbone_lr = float(backbone_lr)

    # Early stopping patience
    early_stop_patience = cfg.get("early_stop_patience")

    return StageConfig(
        name=name,
        enabled=cfg.get("enabled", True),
        freeze_backbone=cfg.get("freeze_backbone", freeze),
        epochs=cfg.get("epochs", epochs),
        learning_rate=learning_rate,
        lr_schedule=(cfg.get("lr_schedule") or "").lower(),
        lr_schedule_params=cfg.get("lr_schedule_params", {}),
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        early_stop_patience=early_stop_patience,
    )


def load_project_settings(config_path: Path, train_cfg_override: Dict | None = None) -> Dict:
    config_path = config_path.resolve()
    base_dir = config_path.parent

    train_cfg = train_cfg_override if train_cfg_override is not None else load_yaml(
        config_path)
    if not train_cfg:
        raise ValueError(f"Config file {config_path} is empty or missing")

    data_cfg = load_yaml(base_dir / "data.yaml")
    model_cfg = load_yaml(base_dir / "model.yaml")

    dataset_defaults = data_cfg.get("dataset", {})
    dataloader_defaults = data_cfg.get("dataloader", {})
    augmentation_defaults = data_cfg.get("augmentation", {})
    normalization_defaults = data_cfg.get("normalization", {})
    paths = data_cfg.get("paths", {})

    data_overrides = train_cfg.get("data", {})
    train_path = data_overrides.get("path", paths.get("train", "./data/train"))
    val_path = data_overrides.get("val_path", paths.get("val", "./data/val"))

    # Get model name first to access model-specific batch_size
    model_name = train_cfg.get("model", model_cfg.get("default_model"))
    if not model_name:
        raise ValueError("No model specified in train config or model.yaml")

    try:
        model_settings = model_cfg["models"][model_name]
    except KeyError as exc:
        raise KeyError(
            f"Model '{model_name}' not found in model.yaml") from exc

    # Resolve batch_size with priority:
    # 1. model_config section in train_cfg (new structure for environment-specific configs)
    # 2. data_overrides.batch_size (explicit override in data section)
    # 3. model_settings.batch_size (from model.yaml)
    # 4. dataloader_defaults.batch_size (from data.yaml)
    # 5. Default fallback: 32

    batch_size = None

    # Check for model_config section (new structure in train_runpod.yaml)
    model_config = train_cfg.get("model_config", {})
    if model_config and model_name in model_config:
        batch_size = model_config[model_name].get("batch_size")

    # If not found, check data_overrides
    if batch_size is None:
        batch_size = data_overrides.get("batch_size")

    # If still None, check model.yaml defaults
    if batch_size is None:
        batch_size = model_settings.get("batch_size")

    # Final fallback
    if batch_size is None:
        batch_size = dataloader_defaults.get("batch_size", 32)

    # Handle string values (in case user wrote "${model_config.${model}.batch_size}" literally)
    if isinstance(batch_size, str) and batch_size.startswith("${"):
        # Try to resolve from model_config
        if model_config and model_name in model_config:
            batch_size = model_config[model_name].get("batch_size", 32)
        else:
            batch_size = 32

    val_batch = data_overrides.get("val_batch_size", batch_size)

    data_settings = {
        "train_path": resolve_path(train_path),
        "val_path": resolve_path(val_path),
        "batch_size": batch_size,
        "val_batch_size": val_batch,
        "num_workers": data_overrides.get("num_workers", dataloader_defaults.get("num_workers", 4)),
        "pin_memory": data_overrides.get("pin_memory", dataloader_defaults.get("pin_memory", True)),
        "drop_last": data_overrides.get("drop_last", dataloader_defaults.get("drop_last", False)),
        "shuffle": data_overrides.get("shuffle", True),
        # Use model's input_size if img_size not explicitly set in data config
        "image_size": data_overrides.get("img_size") or model_settings.get("input_size", dataset_defaults.get("image_size", 224)),
        "normalize_grayscale": dataset_defaults.get("normalize_grayscale", True),
        "mean": normalization_defaults.get("mean", [0.485, 0.456, 0.406]),
        "std": normalization_defaults.get("std", [0.229, 0.224, 0.225]),
        "augmentation": {**augmentation_defaults, **data_overrides.get("augmentation", {})},
    }

    training_cfg = train_cfg.get("training", {})
    stage1_cfg = prepare_stage(
        "Stage 1",
        training_cfg.get("stage1"),
        freeze=True,
        epochs=5,
        lr=1e-4,
    )
    stage2_cfg = prepare_stage(
        "Stage 2",
        training_cfg.get("stage2"),
        freeze=False,
        epochs=45,
        lr=3e-5,
    )

    output_dir = resolve_path(training_cfg.get("output_dir", "./checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get monitoring config
    monitoring_cfg = train_cfg.get("monitoring", {})

    return {
        "train_cfg": train_cfg,
        "data": data_settings,
        "model_name": model_name,
        "model_settings": model_settings,
        "stage1": stage1_cfg,
        "stage2": stage2_cfg,
        "output_dir": output_dir,
        "seed": train_cfg.get("seed", 42),
        "monitoring": monitoring_cfg,
    }


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(data_cfg: Dict, split: str) -> transforms.Compose:
    size = data_cfg["image_size"]
    aug = data_cfg["augmentation"]
    tfms = [transforms.Lambda(lambda img: img.convert("RGB"))]
    tfms.append(transforms.Resize((size, size)))

    if split == "train" and aug.get("enabled", False):
        if aug.get("horizontal_flip", False):
            tfms.append(transforms.RandomHorizontalFlip())
        rotation = aug.get("rotation", 0)
        if rotation:
            tfms.append(transforms.RandomRotation(rotation))
        jitter = aug.get("color_jitter", 0)
        if jitter:
            tfms.append(transforms.ColorJitter(jitter, jitter, jitter, 0))

    tfms.append(transforms.ToTensor())
    tfms.append(transforms.Normalize(
        mean=data_cfg["mean"], std=data_cfg["std"]))
    return transforms.Compose(tfms)


def create_dataloaders(data_cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    train_ds = datasets.ImageFolder(
        str(data_cfg["train_path"]), transform=build_transforms(data_cfg, "train"))
    val_ds = datasets.ImageFolder(
        str(data_cfg["val_path"]), transform=build_transforms(data_cfg, "val"))

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=data_cfg["shuffle"],
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        drop_last=data_cfg["drop_last"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg["val_batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
    )
    return train_loader, val_loader


def freeze_model(model: nn.Module, freeze_backbone: bool) -> None:
    """Freeze or unfreeze model parameters based on stage."""
    if freeze_backbone:
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only the head (classifier)
        if hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
        else:
            # Fallback: try to find classifier/head by common names
            for name, module in model.named_children():
                if name in ['head', 'classifier', 'fc']:
                    for param in module.parameters():
                        param.requires_grad = True
                    break

        # Verify that at least some parameters require grad
        trainable_params = sum(
            1 for p in model.parameters() if p.requires_grad)
        if trainable_params == 0:
            raise RuntimeError(
                "No trainable parameters found after freezing backbone. "
                "Check that model.head exists and has parameters."
            )
    else:
        # Unfreeze all parameters for full fine-tuning
        for param in model.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------
# Build Optimizer
# ---------------------------------------------------------------
def build_optimizer(model: nn.Module, stage: StageConfig):
    """
    Selects Adam for frozen-backbone training, AdamW for full fine-tuning.
    For Stage 2, uses different learning rates for backbone and head to prevent
    destroying pretrained features.
    Only parameters with requires_grad=True are optimized.
    """
    params = list(filter(lambda p: p.requires_grad, model.parameters()))

    if len(params) == 0:
        raise RuntimeError(
            f"No trainable parameters found for {stage.name}. "
            "Check that freeze_model correctly set requires_grad=True for some parameters."
        )

    # Stage 1: backbone frozen → Adam is enough (small #params)
    if stage.freeze_backbone:
        return Adam(params, lr=stage.learning_rate)

    # Stage 2: full fine-tuning → AdamW with separate LRs for backbone and head
    # This prevents destroying pretrained features while allowing head to adapt faster
    weight_decay = getattr(stage, 'weight_decay', 0.01)

    # Get separate learning rates if specified, otherwise use defaults
    # Default: same as stage LR
    head_lr = getattr(stage, 'head_lr', stage.learning_rate)
    # Default: 20% of head LR
    backbone_lr = getattr(stage, 'backbone_lr', stage.learning_rate * 0.2)

    # Separate backbone and head parameters
    backbone_params = []
    head_params = []

    # Identify head parameters (usually the last layer/module)
    if hasattr(model, 'head'):
        head_module = model.head
        head_param_ids = {id(p) for p in head_module.parameters()}

        for p in params:
            if id(p) in head_param_ids:
                head_params.append(p)
            else:
                backbone_params.append(p)
    else:
        # If no explicit head, use last layer as head
        # Get all named modules and find the classifier/head
        all_modules = list(model.named_modules())
        if len(all_modules) > 1:
            # Assume last child module is the head
            last_module_name, last_module = all_modules[-1]
            head_param_ids = {id(p) for p in last_module.parameters()}

            for p in params:
                if id(p) in head_param_ids:
                    head_params.append(p)
                else:
                    backbone_params.append(p)
        else:
            # Fallback: use all params with head_lr
            head_params = params
            backbone_params = []

    # Create parameter groups with different learning rates
    param_groups = []
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': backbone_lr,
            'weight_decay': weight_decay
        })
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': head_lr,
            'weight_decay': weight_decay
        })

    print(f"  Optimizer: AdamW with separate LRs")
    if backbone_params:
        print(
            f"    Backbone LR: {backbone_lr:.2e} ({len(backbone_params)} params)")
    if head_params:
        print(f"    Head LR: {head_lr:.2e} ({len(head_params)} params)")

    return AdamW(param_groups)


# ---------------------------------------------------------------
# Build Scheduler
# ---------------------------------------------------------------
def build_scheduler(
    optimizer,
    stage: StageConfig,
    epochs: int,
    use_warmup: bool = False,
    warmup_epochs: int = 5
):
    """
    Creates a scheduler depending on config:
    - cosine
    - reducelronplateau
    - optional warmup wrapping
    """
    # No scheduler
    if stage.lr_schedule is None:
        return None

    # -------------------------------
    # COSINE ANNEALING
    # -------------------------------
    if stage.lr_schedule == "cosine":
        raw_params = stage.lr_schedule_params or {}
        t_max = raw_params.get("T_max", epochs)
        eta_min = raw_params.get("eta_min", 1e-6)

        t_max = int(t_max)
        eta_min = float(eta_min)

        cosine = CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min
        )

        # Wrap in warmup scheduler if enabled
        if use_warmup:
            return LinearWarmupScheduler(
                optimizer,
                warmup_epochs=warmup_epochs,
                base_scheduler=cosine
            )
        return cosine

    # -------------------------------
    # REDUCE LR ON PLATEAU
    # -------------------------------
    if stage.lr_schedule == "reducelronplateau":
        raw_params = stage.lr_schedule_params or {}

        patience = int(raw_params.get("patience", 5))
        factor = float(raw_params.get("factor", 0.5))
        min_lr = float(raw_params.get("min_lr", 1e-7))
        mode = raw_params.get("mode", "min")  # val_loss typically

        plateau = ReduceLROnPlateau(
            optimizer,
            patience=patience,
            factor=factor,
            min_lr=min_lr,
            mode=mode
        )

        return plateau

    # Nothing matched
    return None


def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, targets)

        if not loss.requires_grad:
            raise RuntimeError(
                "Loss does not require gradients. Check that model outputs are part of computation graph."
            )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return total_loss / total, correct / total if total else 0.0


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return total_loss / total, correct / total if total else 0.0


def save_checkpoint(model, path: Path, stage: StageConfig, epoch: int, val_acc: float) -> None:
    torch.save(
        {
            "stage": stage.name,
            "epoch": epoch,
            "val_accuracy": val_acc,
            "state_dict": model.state_dict(),
        },
        path,
    )


def run_stage(
    model, device, train_loader, val_loader, stage: StageConfig, criterion,
    start_epoch: int, best_acc: float, ckpt_path: Path,
    tensorboard_writer: Optional = None, wandb_run: Optional = None, mlflow_run: Optional = None
) -> Tuple[int, float]:
    if not stage.enabled or stage.epochs <= 0:
        print(f"⏭️  {stage.name} skipped")
        return start_epoch, best_acc

    status = "Freezing" if stage.freeze_backbone else "Unfreezing"
    print(f"\n➡️  {stage.name}: {status} backbone")

    freeze_model(model, stage.freeze_backbone)

    # Check trainable parameters after freezing
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Test forward pass to verify gradients flow (silent check)
    model.train()
    test_input = torch.randn(1, 3, 224, 224).to(device)
    test_output = model(test_input)
    if not test_output.requires_grad:
        raise RuntimeError(
            "Model outputs do not require gradients even though head parameters require grad. "
            "This suggests the forward pass is not using the trainable head."
        )
    total = sum(p.numel() for p in model.parameters())
    print(
        f"  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    optimizer = build_optimizer(model, stage)
    scheduler = build_scheduler(optimizer, stage, stage.epochs)

    # Early stopping configuration
    early_stop_patience = getattr(stage, 'early_stop_patience', None)
    if early_stop_patience is None and not stage.freeze_backbone:
        # Default early stopping for Stage 2: stop if no improvement for 5 epochs
        early_stop_patience = 5
    elif stage.freeze_backbone:
        early_stop_patience = None  # No early stopping for Stage 1

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_val_acc_for_early_stop = best_acc

    current_epoch = start_epoch
    for epoch in range(stage.epochs):
        current_epoch += 1
        train_loss, train_acc = train_one_epoch(
            model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler:
            scheduler.step()

        # Early stopping check (only for Stage 2)
        should_stop_early = False
        if early_stop_patience is not None and not stage.freeze_backbone:
            # Check if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Also check validation accuracy
            if val_acc > best_val_acc_for_early_stop:
                best_val_acc_for_early_stop = val_acc
            elif val_acc <= best_val_acc_for_early_stop:
                # If accuracy didn't improve, count towards early stopping
                pass

            if epochs_without_improvement >= early_stop_patience:
                should_stop_early = True
                print(
                    f"\n⏹️  Early stopping triggered: No improvement in val loss for {early_stop_patience} epochs")
                print(
                    f"   Best val loss: {best_val_loss:.4f}, Best val acc: {best_val_acc_for_early_stop*100:.2f}%")

        print(
            f"{stage.name} | Epoch {epoch+1}/{stage.epochs} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc*100:.2f}% | "
            f"Val Loss {val_loss:.4f} Acc {val_acc*100:.2f}% | LR {current_lr:.2e}"
        )

        # Log to TensorBoard
        if tensorboard_writer:
            try:
                tensorboard_writer.add_scalar(
                    f"{stage.name}/Train/Loss", train_loss, current_epoch)
                tensorboard_writer.add_scalar(
                    f"{stage.name}/Train/Accuracy", train_acc, current_epoch)
                tensorboard_writer.add_scalar(
                    f"{stage.name}/Val/Loss", val_loss, current_epoch)
                tensorboard_writer.add_scalar(
                    f"{stage.name}/Val/Accuracy", val_acc, current_epoch)
                tensorboard_writer.add_scalar(
                    f"{stage.name}/LearningRate", current_lr, current_epoch)
            except Exception as e:
                print(f"⚠️  TensorBoard logging failed: {e}")

        # Log to Weights & Biases
        if wandb_run:
            try:
                wandb_run.log({
                    f"{stage.name}/train_loss": train_loss,
                    f"{stage.name}/train_acc": train_acc,
                    f"{stage.name}/val_loss": val_loss,
                    f"{stage.name}/val_acc": val_acc,
                    f"{stage.name}/learning_rate": current_lr,
                    "epoch": current_epoch,
                })
            except Exception as e:
                print(f"⚠️  W&B logging failed: {e}")

        # Log to MLflow
        if mlflow_run:
            try:
                mlflow.log_metrics({
                    f"{stage.name}_train_loss": train_loss,
                    f"{stage.name}_train_acc": train_acc,
                    f"{stage.name}_val_loss": val_loss,
                    f"{stage.name}_val_acc": val_acc,
                    f"{stage.name}_learning_rate": current_lr,
                }, step=current_epoch)
            except Exception as e:
                print(f"⚠️  MLflow logging failed: {e}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, ckpt_path, stage, current_epoch, val_acc)
            print(f"💾 Saved new best checkpoint (Val Acc {val_acc*100:.2f}%)")

            # Log best checkpoint
            if wandb_run:
                try:
                    wandb_run.log({"best_val_acc": best_acc,
                                  "best_epoch": current_epoch})
                except Exception as e:
                    print(f"⚠️  W&B best checkpoint logging failed: {e}")
            if mlflow_run:
                try:
                    mlflow.log_metric(
                        "best_val_acc", best_acc, step=current_epoch)
                except Exception as e:
                    print(f"⚠️  MLflow best checkpoint logging failed: {e}")
                mlflow.log_artifact(str(ckpt_path), "checkpoints")

        # Early stopping: break if triggered
        if should_stop_early:
            print(f"\n⏹️  Stopping training early due to no improvement")
            break

    return current_epoch, best_acc


def get_all_models(config_path: Path) -> list[str]:
    """Get list of all available models from model.yaml"""
    base_dir = config_path.parent
    model_cfg = load_yaml(base_dir / "model.yaml")
    if not model_cfg or "models" not in model_cfg:
        raise ValueError("No models found in model.yaml")
    return list(model_cfg["models"].keys())


def train_single_model(config_path: Path, device: str | None, model_name: str | None = None):
    """Train a single model. If model_name is None, uses the model from config."""
    # Verify environment setup (works both locally and on RunPod/Docker)
    home_dir = os.getenv("HOME", "~")
    is_runpod = os.getenv(
        "HOME") == "/workspace" or os.path.exists("/workspace")

    if is_runpod:
        print("🌐 Running in RunPod/Docker environment")
        print(f"   HOME={home_dir}")
    else:
        print("💻 Running in local environment")
        print(f"   HOME={home_dir}")

    # Override model if specified
    train_cfg = load_yaml(config_path)
    if model_name:
        train_cfg["model"] = model_name

    settings = load_project_settings(config_path, train_cfg_override=train_cfg)
    set_seed(settings["seed"])

    device_obj = torch.device(device or (
        "cuda" if torch.cuda.is_available() else "cpu"))
    data_cfg = settings["data"]

    # Download dataset if needed (uses KAGGLE_USERNAME and KAGGLE_KEY env vars)
    download_dataset_if_needed(data_cfg["train_path"], data_cfg["val_path"])

    train_loader, val_loader = create_dataloaders(data_cfg)

    print("\n🔧 Training Configuration")
    print(f"  Device: {device_obj}")
    print(f"  Seed: {settings['seed']}")
    model_info = settings["model_settings"]
    print(f"  Model: {model_info['name']}")
    print(f"  Model settings: {model_info}")
    print(f"  Train path: {data_cfg['train_path']}")
    print(f"  Val path: {data_cfg['val_path']}")
    print(f"  Batch size: {data_cfg['batch_size']}")
    print(f"  Val batch size: {data_cfg['val_batch_size']}")
    print(f"  Image size: {data_cfg['image_size']}")
    print(f"  Mean: {data_cfg['mean']}")
    print(f"  Std: {data_cfg['std']}")
    print(f"  Augmentation: {data_cfg['augmentation']}")
    print(f"  Stage 1: {settings['stage1']}")
    print(f"  Stage 2: {settings['stage2']}")
    model = get_model(
        model_info["name"],
        model_info["num_classes"],
        pretrained=model_info.get("pretrained", True),
    ).to(device_obj)

    # Verify model structure
    print(f"\n🔍 Model Structure Check:")
    print(f"  Has 'head' attribute: {hasattr(model, 'head')}")
    if hasattr(model, 'head'):
        print(f"  Head: {model.head}")
    else:
        print(
            f"  Available attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        # Try to find classifier/head by inspecting children
        for name, module in model.named_children():
            print(f"    {name}: {type(module)}")

    # Build loss function with label smoothing if specified
    loss_cfg = settings.get("loss", {})
    label_smoothing = loss_cfg.get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        print(
            f"  Loss: CrossEntropyLoss with label_smoothing={label_smoothing}")
    else:
        print(f"  Loss: CrossEntropyLoss")
    ckpt_path = settings["output_dir"] / "best_model.pth"
    last_model_path = settings["output_dir"] / "last_model.pth"

    # Initialize logging
    monitoring_cfg = settings.get("monitoring", {})
    tensorboard_writer = None
    wandb_run = None
    mlflow_run = None

    # TensorBoard
    if TENSORBOARD_AVAILABLE:
        try:
            tb_log_dir = monitoring_cfg.get("tensorboard_log_dir", "./runs")
            tb_log_dir = resolve_path(tb_log_dir)
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            tensorboard_writer = SummaryWriter(str(tb_log_dir))
            print(f"📊 TensorBoard logging to: {tb_log_dir}")
        except Exception as e:
            print(f"⚠️  TensorBoard logging disabled: {e}")
            tensorboard_writer = None

    # Weights & Biases
    if WANDB_AVAILABLE:
        try:
            wandb_cfg = monitoring_cfg.get("wandb", {})
            if wandb_cfg.get("project"):
                # Automatically login using WANDB_API_KEY if available (only if not already logged in)
                wandb_api_key = os.getenv("WANDB_API_KEY")
                if wandb_api_key:
                    try:
                        wandb.login(key=wandb_api_key, relogin=True)
                        print(f"✅ W&B automatically logged in using WANDB_API_KEY")
                    except Exception:
                        # Already logged in, continue
                        pass
                else:
                    print("ℹ️  WANDB_API_KEY not in environment.")
                    print(
                        "   W&B will try to use cached credentials or prompt for login.")
                    print(
                        "   (Set WANDB_API_KEY environment variable for automatic login)")

                # Create unique run name with model name
                run_name = wandb_cfg.get("name")
                if not run_name:
                    run_name = f"{settings['model_name']}-training"

                # Add model name to tags if not already present
                tags = wandb_cfg.get("tags", [])
                if settings['model_name'] not in tags:
                    tags = tags + [settings['model_name']]

                wandb_run = wandb.init(
                    project=wandb_cfg.get("project"),
                    entity=wandb_cfg.get("entity"),
                    name=run_name,
                    tags=tags,
                    notes=wandb_cfg.get("notes", ""),
                    config={
                        "model": settings["model_name"],
                        "model_settings": model_info,
                        "data": data_cfg,
                        "stage1": settings["stage1"].__dict__,
                        "stage2": settings["stage2"].__dict__,
                        "seed": settings["seed"],
                    }
                )
                print(
                    f"🔮 Weights & Biases logging enabled: {wandb_cfg.get('project')}")
        except Exception as e:
            print(f"⚠️  Weights & Biases logging disabled: {e}")
            wandb_run = None

    # MLflow
    if MLFLOW_AVAILABLE:
        try:
            mlflow_uri = monitoring_cfg.get("mlflow_tracking_uri", "./mlruns")
            mlflow.set_tracking_uri(mlflow_uri)
            experiment_name = monitoring_cfg.get(
                "mlflow_experiment_name", "lungs-disease")
            mlflow.set_experiment(experiment_name)
            mlflow_run = mlflow.start_run(
                run_name=f"{settings['model_name']}-training")
            mlflow.log_params({
                "model": settings["model_name"],
                "model_name": model_info["name"],
                "num_classes": model_info["num_classes"],
                "pretrained": model_info.get("pretrained", True),
                "batch_size": data_cfg["batch_size"],
                "image_size": data_cfg["image_size"],
                "stage1_epochs": settings["stage1"].epochs,
                "stage1_lr": settings["stage1"].learning_rate,
                "stage2_epochs": settings["stage2"].epochs,
                "stage2_lr": settings["stage2"].learning_rate,
                "seed": settings["seed"],
            })
            print(f"📈 MLflow logging enabled: {experiment_name}")
        except Exception as e:
            print(f"⚠️  MLflow logging disabled: {e}")
            mlflow_run = None

    total_epochs = 0
    best_accuracy = 0.0

    for stage in (settings["stage1"], settings["stage2"]):
        total_epochs, best_accuracy = run_stage(
            model,
            device_obj,
            train_loader,
            val_loader,
            stage,
            criterion,
            total_epochs,
            best_accuracy,
            ckpt_path,
            tensorboard_writer,
            wandb_run,
            mlflow_run,
        )

    torch.save(model.state_dict(), last_model_path)

    # Log final model
    if wandb_run:
        try:
            wandb_run.log_artifact(str(ckpt_path))
            wandb_run.log_artifact(str(last_model_path))
        except Exception as e:
            print(f"⚠️  Failed to log artifacts to W&B: {e}")

    if mlflow_run:
        try:
            mlflow.log_artifact(str(last_model_path), "models")
            mlflow.pytorch.log_model(model, "model")
        except Exception as e:
            print(f"⚠️  Failed to log model to MLflow: {e}")

    # Close loggers
    if tensorboard_writer:
        try:
            tensorboard_writer.close()
        except Exception as e:
            print(f"⚠️  Failed to close TensorBoard: {e}")

    if wandb_run:
        try:
            wandb_run.finish()
        except Exception as e:
            print(f"⚠️  Failed to finish W&B run: {e}")

    if mlflow_run:
        try:
            mlflow.end_run()
        except Exception as e:
            print(f"⚠️  Failed to end MLflow run: {e}")

    print("\n✅ Training finished")
    print(f"Best validation accuracy: {best_accuracy*100:.2f}%")
    print(f"Best checkpoint: {ckpt_path}")
    print(f"Last model: {last_model_path}")

    return best_accuracy


def main(config_path: Path, device: str | None):
    """Main function that trains all models or a single model based on config."""
    # Check if we should train all models
    train_cfg = load_yaml(config_path)
    train_all_models = train_cfg.get("train_all_models", False)

    # Login to wandb once if API key is available (for all models training)
    if WANDB_AVAILABLE and train_all_models:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            try:
                wandb.login(key=wandb_api_key, relogin=True)
                print(
                    f"✅ W&B automatically logged in using WANDB_API_KEY (for all models)")
            except Exception as e:
                print(f"⚠️  W&B login failed: {e}")

    if train_all_models:
        # Train all models from model.yaml
        all_models = get_all_models(config_path)
        print(
            f"\n🚀 Training all {len(all_models)} models: {', '.join(all_models)}")
        print("=" * 80)

        results = {}
        for idx, model_name in enumerate(all_models, 1):
            print(f"\n{'=' * 80}")
            print(f"📦 Model {idx}/{len(all_models)}: {model_name}")
            print(f"{'=' * 80}")

            try:
                best_acc = train_single_model(config_path, device, model_name)
                results[model_name] = best_acc
                print(
                    f"\n✅ {model_name} completed: {best_acc*100:.2f}% accuracy")
            except Exception as e:
                print(f"\n❌ {model_name} failed: {e}")
                results[model_name] = None
                import traceback
                traceback.print_exc()

        # Print summary
        print(f"\n{'=' * 80}")
        print("📊 Training Summary")
        print(f"{'=' * 80}")
        for model_name, acc in results.items():
            if acc is not None:
                print(f"  {model_name:30s}: {acc*100:6.2f}%")
            else:
                print(f"  {model_name:30s}: FAILED")

        if results:
            successful = {k: v for k, v in results.items() if v is not None}
            if successful:
                best_model = max(successful.items(), key=lambda x: x[1])
                print(
                    f"\n🏆 Best model: {best_model[0]} ({best_model[1]*100:.2f}%)")
    else:
        # Train single model (original behavior)
        train_single_model(config_path, device, None)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage medical fine-tuning")
    parser.add_argument("--config", type=str,
                        default=str(DEFAULT_CONFIG), help="Path to training config (default: train_runpod.yaml)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device id (e.g., cuda:0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.config), args.device)
