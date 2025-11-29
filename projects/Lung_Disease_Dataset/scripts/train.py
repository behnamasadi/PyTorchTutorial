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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train.yaml"

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


def prepare_stage(name: str, stage_cfg: Dict, *, freeze: bool, epochs: int, lr: float) -> StageConfig:
    cfg = stage_cfg or {}
    return StageConfig(
        name=name,
        enabled=cfg.get("enabled", True),
        freeze_backbone=cfg.get("freeze_backbone", freeze),
        epochs=cfg.get("epochs", epochs),
        learning_rate=cfg.get("learning_rate", lr),
        lr_schedule=(cfg.get("lr_schedule") or "").lower(),
        lr_schedule_params=cfg.get("lr_schedule_params", {}),
    )


def load_project_settings(config_path: Path) -> Dict:
    config_path = config_path.resolve()
    base_dir = config_path.parent

    train_cfg = load_yaml(config_path)
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
        raise ValueError("No model specified in train.yaml or model.yaml")

    try:
        model_settings = model_cfg["models"][model_name]
    except KeyError as exc:
        raise KeyError(
            f"Model '{model_name}' not found in model.yaml") from exc

    # Use model-specific batch_size with fallback to data_overrides or dataloader_defaults
    batch_size = (
        model_settings.get("batch_size") or
        data_overrides.get("batch_size") or
        dataloader_defaults.get("batch_size", 32)
    )
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
        "image_size": data_overrides.get("img_size", dataset_defaults.get("image_size", 224)),
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

    # Stage 2: full fine-tuning → AdamW is correct (weight decay)
    # Default weight_decay of 0.01 is standard for AdamW
    weight_decay = getattr(stage, 'weight_decay', 0.01)
    return AdamW(
        params,
        lr=stage.learning_rate,
        weight_decay=weight_decay
    )


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

        # Debug first batch
        if batch_idx == 0:
            print(f"  Debug batch 0:")
            print(f"    Outputs requires_grad: {outputs.requires_grad}")
            print(f"    Outputs grad_fn: {outputs.grad_fn}")
            # Check if any head parameters require grad
            head_trainable = any(
                p.requires_grad for p in model.head.parameters())
            print(f"    Head has trainable params: {head_trainable}")

        loss = criterion(outputs, targets)

        if batch_idx == 0:
            print(f"    Loss requires_grad: {loss.requires_grad}")
            print(f"    Loss grad_fn: {loss.grad_fn}")

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

    # Debug: Check model structure before freezing
    print(f"  Model has 'head' attribute: {hasattr(model, 'head')}")
    if hasattr(model, 'head'):
        print(f"  Head type: {type(model.head)}")
        head_params = sum(p.numel() for p in model.head.parameters())
        print(f"  Head parameters: {head_params:,}")

    freeze_model(model, stage.freeze_backbone)

    # Debug: Check trainable parameters after freezing
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Debug: Test forward pass to verify gradients flow
    model.train()
    test_input = torch.randn(1, 3, 224, 224).to(device)
    test_output = model(test_input)
    print(f"  Test forward output requires_grad: {test_output.requires_grad}")
    print(f"  Test forward output grad_fn: {test_output.grad_fn}")
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

    return current_epoch, best_acc


def main(config_path: Path, device: str | None):
    settings = load_project_settings(config_path)
    set_seed(settings["seed"])

    device_obj = torch.device(device or (
        "cuda" if torch.cuda.is_available() else "cpu"))
    data_cfg = settings["data"]
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

    criterion = nn.CrossEntropyLoss()
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
                wandb_run = wandb.init(
                    project=wandb_cfg.get("project"),
                    entity=wandb_cfg.get("entity"),
                    name=wandb_cfg.get(
                        "name") or f"{settings['model_name']}-training",
                    tags=wandb_cfg.get("tags", []),
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage medical fine-tuning")
    parser.add_argument("--config", type=str,
                        default=str(DEFAULT_CONFIG), help="Path to train.yaml")
    parser.add_argument("--device", type=str, default=None,
                        help="Device id (e.g., cuda:0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.config), args.device)
