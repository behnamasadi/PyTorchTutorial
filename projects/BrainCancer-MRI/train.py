import argparse
import yaml
import os
import time
from utils.path_utils import get_project_root

from models.model import get_model
from data.dataset import load_datasets
from utils.helpers import save_checkpoint, calculate_mean_std
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torch.amp import GradScaler
from torch.amp import autocast
from torch.optim.lr_scheduler import OneCycleLR
import psutil
import GPUtil

# Monitoring imports
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch
import wandb


def get_hardware_info():
    """Get current hardware utilization info"""
    info = {}

    # CPU info
    info['cpu_percent'] = psutil.cpu_percent(interval=1)
    info['cpu_count'] = psutil.cpu_count()

    # Memory info
    memory = psutil.virtual_memory()
    info['memory_percent'] = memory.percent
    info['memory_available_gb'] = memory.available / (1024**3)

    # GPU info
    if torch.cuda.is_available():
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # First GPU
                info['gpu_name'] = gpu.name
                info['gpu_memory_percent'] = gpu.memoryUtil * 100
                info['gpu_memory_used_mb'] = gpu.memoryUsed
                info['gpu_memory_total_mb'] = gpu.memoryTotal
                info['gpu_temperature'] = gpu.temperature
                info['gpu_utilization'] = gpu.load * 100
        except:
            info['gpu_available'] = False
    else:
        info['gpu_available'] = False

    return info


def main(config_path):
    # Resolve absolute path to config file
    if not os.path.isabs(config_path):
        # Resolve relative to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve all output paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Fix monitoring paths and make them model-specific
    selected_model = config['model']
    for key in ['tensorboard_log_dir', 'mlflow_tracking_uri']:
        if key in config['monitoring'] and not os.path.isabs(config['monitoring'][key]):
            # Remove leading ./ if present and join with script directory
            rel_path = config['monitoring'][key].lstrip('./')
            base_path = os.path.join(script_dir, rel_path)
            # Make model-specific
            if key == 'tensorboard_log_dir':
                config['monitoring'][key] = os.path.join(
                    base_path, f"{selected_model}_logs")
            else:
                config['monitoring'][key] = base_path

    # Fix training output path and make it model-specific
    if not os.path.isabs(config['train']['output_dir']):
        rel_path = config['train']['output_dir'].lstrip('./')
        base_output_dir = os.path.join(script_dir, rel_path)
    else:
        base_output_dir = config['train']['output_dir']

    # Create model-specific output directory
    selected_model = config['model']
    model_output_dir = os.path.join(
        base_output_dir, f"{selected_model}_outputs")
    config['train']['output_dir'] = model_output_dir

    # Fix dataset path
    if not os.path.isabs(config['dataset']['path']):
        rel_path = config['dataset']['path'].lstrip('./')
        config['dataset']['path'] = os.path.join(script_dir, rel_path)

    # Get selected model configuration
    selected_model = config['model']
    model_config = config['models'][selected_model]

    weights = eval(model_config['weights']
                   ) if model_config['weights'] else None

    model, classifier_params = get_model(
        model_config['name'],
        model_config['num_classes'],
        weights
    )

    # Load datasets without normalization (back to basics)
    print("📊 Loading datasets without normalization...")

    # Use model-specific image size if available
    if 'img_size' in model_config:
        original_img_size = config['dataset']['img_size']
        config['dataset']['img_size'] = model_config['img_size']
        print(
            f"🖼️  Using model-specific image size: {model_config['img_size']}x{model_config['img_size']}")

    train_ds, val_ds, _ = load_datasets(config, mean=None, std=None)

    # Use model-specific batch size if available, otherwise use global setting
    batch_size = model_config.get(
        'batch_size', config['dataset']['batch_size'])

    # Set up optimized data loading
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': config['dataset']['num_workers'],
        'pin_memory': config['dataset']['pin_memory'],
        'prefetch_factor': config['dataset'].get('prefetch_factor', 2),
        'persistent_workers': config['dataset'].get('persistent_workers', True)
    }

    train_loader = DataLoader(train_ds, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dataloader_kwargs)

    # Use Adamax for xception_medical (as per Kaggle solution), AdamW for others
    if selected_model == 'xception_medical':
        optimizer = optim.Adamax(classifier_params, lr=model_config['lr'])
        print(
            f"🔧 Using Adamax optimizer (Kaggle solution) with LR: {model_config['lr']}")
    else:
        optimizer = optim.AdamW(classifier_params, lr=model_config['lr'])
        print(f"🔧 Using AdamW optimizer with LR: {model_config['lr']}")
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler with warmup
    scheduler = OneCycleLR(
        optimizer,
        max_lr=model_config['lr'],
        epochs=config['train']['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )

    # Performance optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if torch.cuda.is_available() and config.get('performance', {}).get('benchmark_cudnn', True):
        torch.backends.cudnn.benchmark = True

    # Mixed precision training
    use_amp = config.get('performance', {}).get(
        'mixed_precision', False) and torch.cuda.is_available()
    scaler = GradScaler('cuda') if use_amp else None

    # Memory format optimization
    if config.get('performance', {}).get('channels_last', False) and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)

    # Model compilation (PyTorch 2.0+)
    if config.get('performance', {}).get('compile_model', False):
        try:
            model = torch.compile(model)
            print("✅ Model compiled for faster training")
        except Exception as e:
            print(f"⚠️  Model compilation failed: {e}")

    # Gradient clipping
    max_grad_norm = config.get('performance', {}).get('max_grad_norm', None)

    # Setup monitoring
    # TensorBoard
    os.makedirs(config['monitoring']['tensorboard_log_dir'], exist_ok=True)
    writer = SummaryWriter(config['monitoring']['tensorboard_log_dir'])

    # Weights & Biases
    wandb_config = config['monitoring']['wandb']
    wandb.init(
        project=wandb_config['project'],
        entity=wandb_config['entity'],
        name=wandb_config['name'],
        tags=wandb_config['tags'],
        notes=wandb_config['notes'],
        config={
            **model_config,
            **config['dataset'],
            **config['train'],
            'model_type': selected_model
        }
    )

    # MLflow
    # Setup MLflow with model-specific experiment
    mlflow.set_tracking_uri(config['monitoring']['mlflow_tracking_uri'])
    model_experiment_name = f"{config['monitoring']['mlflow_experiment_name']}-{selected_model}"
    mlflow.set_experiment(model_experiment_name)

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "model_name": model_config['name'],
            "model_type": selected_model,
            "num_classes": model_config['num_classes'],
            "learning_rate": model_config['lr'],
            "batch_size": config['dataset']['batch_size'],
            "epochs": config['train']['epochs'],
            "img_size": config['dataset']['img_size'],
            "train_split": config['dataset']['train_split'],
            "val_split": config['dataset']['val_split'],
            "augmentation": config['transform']['augmentation']
        })

        # Print training configuration
        hw_info = get_hardware_info()

        print("="*60)
        print("🚀 Starting Brain Cancer MRI Training")
        print("="*60)
        print(
            f"📊 Dataset: {len(train_ds)} train, {len(val_ds)} validation samples")
        print(
            f"🏗️  Model: {selected_model} ({model_config['name']}) with {model_config['num_classes']} classes")
        print(f"📦 Batch size: {batch_size} (model-optimized)")
        print(f"📈 Learning rate: {model_config['lr']}")
        print(f"🔄 Epochs: {config['train']['epochs']}")
        print(f"🎯 Device: {device}")

        # Hardware utilization
        print(
            f"💻 CPU: {hw_info['cpu_count']} cores @ {hw_info['cpu_percent']:.1f}%")
        print(f"🧠 RAM: {hw_info['memory_percent']:.1f}% used")

        if hw_info.get('gpu_available', False):
            print(f"🚀 GPU: {hw_info['gpu_name']}")
            print(
                f"📊 GPU Usage: {hw_info['gpu_utilization']:.1f}% | Memory: {hw_info['gpu_memory_percent']:.1f}% ({hw_info['gpu_memory_used_mb']:.0f}/{hw_info['gpu_memory_total_mb']:.0f}MB)")

        # Performance features
        if use_amp:
            print("⚡ Mixed Precision: Enabled")
        if config.get('performance', {}).get('compile_model', False):
            print("🔥 Model Compilation: Enabled")

        print(f"📈 TensorBoard: {config['monitoring']['tensorboard_log_dir']}")
        print(f"📊 MLflow: {config['monitoring']['mlflow_experiment_name']}")
        print(f"🔮 Wandb: {config['monitoring']['wandb']['project']}")
        print("="*60)

        # Early stopping variables
        best_val_acc = 0.0
        patience_counter = 0
        patience = config['train'].get('patience', 10)
        min_delta = config['train'].get('min_delta', 0.001)

        print(f"⏰ Early stopping: patience={patience}, min_delta={min_delta}")

        for epoch in range(config['train']['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            epoch_start_time = time.time()

            print(f"\n📚 Epoch {epoch+1}/{config['train']['epochs']}")
            print("-" * 50)

            for batch_idx, (x, y) in enumerate(train_loader):
                # Move data to device
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # Memory format optimization
                if config.get('performance', {}).get('channels_last', False):
                    x = x.to(memory_format=torch.channels_last)

                optimizer.zero_grad()

                # Mixed precision forward pass
                if use_amp:
                    with autocast('cuda'):
                        preds = model(x)
                        loss = criterion(preds, y)

                    # Mixed precision backward pass
                    scaler.scale(loss).backward()

                    # Gradient clipping
                    if max_grad_norm:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm)

                    scaler.step(optimizer)
                    scaler.update()

                    # Update learning rate scheduler
                    scheduler.step()
                else:
                    preds = model(x)
                    loss = criterion(preds, y)
                    loss.backward()

                    # Calculate gradient norm before clipping
                    total_norm = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)

                    # Gradient clipping
                    if max_grad_norm:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm)

                    optimizer.step()

                    # Update learning rate scheduler
                    scheduler.step()

                # Check for NaN values and debug
                loss_value = loss.item()
                if torch.isnan(loss) or torch.isinf(loss):
                    print(
                        f"🚨 NaN/Inf detected at epoch {epoch+1}, batch {batch_idx+1}")
                    print(f"  Loss value: {loss_value}")
                    print(
                        f"  Model output range: [{preds.min().item():.4f}, {preds.max().item():.4f}]")
                    print(
                        f"  Input range: [{x.min().item():.4f}, {x.max().item():.4f}]")

                    # Check gradients
                    total_norm = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    print(f"  Gradient norm: {total_norm:.4f}")

                    # Exit training to prevent further corruption
                    raise ValueError(
                        "NaN/Inf loss detected - stopping training")

                # Calculate metrics
                train_loss += loss_value
                _, predicted = torch.max(preds.data, 1)
                train_total += y.size(0)
                train_correct += (predicted == y).sum().item()

                # Print progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    current_loss = train_loss / (batch_idx + 1)
                    current_acc = 100. * train_correct / train_total
                    print(f"  Batch {batch_idx+1}/{len(train_loader)} | "
                          f"Loss: {current_loss:.4f} | "
                          f"Acc: {current_acc:.2f}%")

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for x, y in val_loader:
                    # Move data to device
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    # Memory format optimization
                    if config.get('performance', {}).get('channels_last', False):
                        x = x.to(memory_format=torch.channels_last)

                    # Mixed precision validation
                    if use_amp:
                        with autocast('cuda'):
                            preds = model(x)
                            loss = criterion(preds, y)
                    else:
                        preds = model(x)
                        loss = criterion(preds, y)

                    val_loss += loss.item()
                    _, predicted = torch.max(preds.data, 1)
                    val_total += y.size(0)
                    val_correct += (predicted == y).sum().item()

            # Calculate epoch metrics
            epoch_time = time.time() - epoch_start_time
            train_loss_avg = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss_avg = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total

            # Early stopping logic
            val_acc_decimal = val_acc / 100.0  # Convert to decimal for comparison
            if val_acc_decimal > best_val_acc + min_delta:
                best_val_acc = val_acc_decimal
                patience_counter = 0
                # Save best model
                best_model_path = os.path.join(
                    config['train']['output_dir'], 'best_model.pth')
                os.makedirs(config['train']['output_dir'], exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc_decimal,
                    'config': config
                }, best_model_path)
                print(
                    f"💾 New best model saved! Val Acc: {val_acc:.2f}% (improvement: {(val_acc_decimal - (best_val_acc - min_delta))*100:.3f}%)")
            else:
                patience_counter += 1
                print(
                    f"⏰ No improvement for {patience_counter}/{patience} epochs (best: {best_val_acc*100:.2f}%)")

                if patience_counter >= patience:
                    print(
                        f"🛑 Early stopping triggered! No improvement for {patience} epochs.")
                    print(
                        f"🏆 Best validation accuracy: {best_val_acc*100:.2f}%")
                    break

            # Validate metrics for NaN/Inf
            if torch.isnan(torch.tensor(train_loss_avg)) or torch.isnan(torch.tensor(val_loss_avg)):
                print("🚨 NaN loss detected in epoch summary - stopping training")
                print(f"  Train loss: {train_loss_avg}")
                print(f"  Val loss: {val_loss_avg}")
                break

            if torch.isinf(torch.tensor(train_loss_avg)) or torch.isinf(torch.tensor(val_loss_avg)):
                print("🚨 Infinite loss detected in epoch summary - stopping training")
                print(f"  Train loss: {train_loss_avg}")
                print(f"  Val loss: {val_loss_avg}")
                break

            # Log to TensorBoard
            writer.add_scalar('Loss/Train', train_loss_avg, epoch)
            writer.add_scalar('Loss/Validation', val_loss_avg, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            writer.add_scalar('Time/Epoch', epoch_time, epoch)

            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_loss_avg,
                'val_loss': val_loss_avg,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'epoch_time': epoch_time
            }, step=epoch)

            # Get hardware metrics
            hw_metrics = get_hardware_info()

            # Log to Wandb
            wandb_metrics = {
                'epoch': epoch,
                'train_loss': train_loss_avg,
                'val_loss': val_loss_avg,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'epoch_time': epoch_time,
                'learning_rate': model_config['lr'],
                'cpu_percent': hw_metrics.get('cpu_percent', 0),
                'memory_percent': hw_metrics.get('memory_percent', 0)
            }

            # Add GPU metrics if available
            if hw_metrics.get('gpu_available', False):
                wandb_metrics.update({
                    'gpu_utilization': hw_metrics.get('gpu_utilization', 0),
                    'gpu_memory_percent': hw_metrics.get('gpu_memory_percent', 0),
                    'gpu_temperature': hw_metrics.get('gpu_temperature', 0)
                })

            wandb.log(wandb_metrics)

            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"  🕐 Time: {epoch_time:.2f}s")
            print(
                f"  📉 Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.2f}%")
            print(
                f"  📊 Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.2f}%")

            # Save checkpoint
            if epoch % config['train']['save_every'] == 0:
                print(f"💾 Saving checkpoint at epoch {epoch+1}")
                save_checkpoint(model, epoch, config)

        # Log final model to MLflow
        mlflow.pytorch.log_model(model, "model")

        # Log model to wandb
        wandb.save("model.pth")

        # Close monitoring
        writer.close()
        wandb.finish()

        print("\n" + "="*60)
        print("🎉 Training completed successfully!")
        print(
            f"📈 TensorBoard logs: {config['monitoring']['tensorboard_log_dir']}")
        print(
            f"📊 MLflow experiment: {config['monitoring']['mlflow_experiment_name']}")
        print(f"🔮 Wandb project: {config['monitoring']['wandb']['project']}")
        print("🔧 To view TensorBoard: tensorboard --logdir runs")
        print("🔧 To view MLflow: mlflow ui")
        print("🔧 To view Wandb: wandb.ai")
        print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Brain Cancer MRI Classification Training')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file (default: config/config.yaml)')
    parser.add_argument('--model', type=str, default=None,
                        help='Override model selection (resnet18, resnet50, swin_t, swin_s, efficientnet_b0, vit_b_16)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')

    args = parser.parse_args()

    # Load config and apply overrides
    if args.model or args.epochs or args.batch_size or args.lr:
        import yaml
        config_path = args.config
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Apply overrides
        if args.model:
            config['model'] = args.model
        if args.epochs:
            config['train']['epochs'] = args.epochs
        if args.batch_size:
            config['dataset']['batch_size'] = args.batch_size
        if args.lr:
            selected_model = config['model']
            config['models'][selected_model]['lr'] = args.lr

        # Save temporary config
        temp_config_path = 'temp_config.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)

        main(temp_config_path)

        # Clean up
        os.remove(temp_config_path)
    else:
        main(args.config)
