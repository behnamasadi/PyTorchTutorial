import argparse
import yaml
import os
import time
import sys
import numpy as np

# Add the project directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

try:
    from utils.path_utils import get_project_root
    from models.model import get_model
    from data.dataset import load_datasets
    from utils.helpers import save_checkpoint, calculate_mean_std

    # Try to import pre-computed normalization constants
    try:
        from normalization_constants import NORMALIZATION_MEAN, NORMALIZATION_STD
        USE_PRECOMPUTED_STATS = True
        print("✅ Using pre-computed normalization statistics")
    except ImportError:
        USE_PRECOMPUTED_STATS = False
        print("⚠️  Pre-computed normalization statistics not found, will compute during training")
        print("💡 Run 'python precompute_normalization.py' to pre-compute statistics")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running the script from the correct directory")
    print("   Expected: src/projects/BrainCancer-MRI/")
    print(f"   Current: {os.getcwd()}")
    sys.exit(1)

try:
    print("🔍 Attempting to import torch...")
    import torch
    print(f"✅ torch imported successfully, version: {torch.__version__}")

    print("🔍 Attempting to import torch.utils.data...")
    from torch.utils.data import DataLoader
    print("✅ torch.utils.data imported successfully")

    print("🔍 Attempting to import torch.optim...")
    import torch.optim as optim
    print("✅ torch.optim imported successfully")

    print("🔍 Attempting to import torch.nn...")
    import torch.nn as nn
    print("✅ torch.nn imported successfully")

    print("🔍 Attempting to import torchvision.models...")
    import torchvision.models as models
    print("✅ torchvision.models imported successfully")

    print("🔍 Attempting to import torch.amp...")
    from torch.amp import GradScaler
    from torch.amp import autocast
    print("✅ torch.amp imported successfully")

    print("🔍 Attempting to import torch.optim.lr_scheduler...")
    from torch.optim.lr_scheduler import OneCycleLR
    print("✅ torch.optim.lr_scheduler imported successfully")

    print("🔍 Attempting to import torch.utils.tensorboard...")
    from torch.utils.tensorboard import SummaryWriter
    print("✅ torch.utils.tensorboard imported successfully")

    print("🎉 All PyTorch imports successful!")
except ImportError as e:
    print(f"❌ PyTorch import error: {e}")
    print("💡 Make sure PyTorch is installed: pip install torch torchvision")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error during PyTorch import: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    import psutil
    import GPUtil
except ImportError as e:
    print(f"❌ System monitoring import error: {e}")
    print("💡 Install required packages: pip install psutil GPUtil")
    sys.exit(1)

# Monitoring imports
try:
    import mlflow
    import mlflow.pytorch
    import wandb
except ImportError as e:
    print(f"❌ Monitoring import error: {e}")
    print("💡 Install monitoring packages: pip install mlflow wandb")
    sys.exit(1)


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


def main(config_path, grayscale=False):
    print("🚀 Starting main training function...")

    # Ensure global imports are available in this function scope
    global torch, mlflow, wandb

    # Resolve absolute path to config file
    if not os.path.isabs(config_path):
        # Resolve relative to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)

    print(f"📁 Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("✅ Configuration loaded successfully")

    # Get grayscale setting from config if not explicitly passed
    if not grayscale:
        grayscale = config.get('dataset', {}).get('grayscale', False)

    # Resolve all output paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Fix monitoring paths and make them model-specific
    selected_model = config['model']

    # Handle tensorboard log directory (file path)
    if 'tensorboard_log_dir' in config['monitoring'] and not os.path.isabs(config['monitoring']['tensorboard_log_dir']):
        rel_path = config['monitoring']['tensorboard_log_dir'].lstrip('./')
        base_path = os.path.join(script_dir, rel_path)
        config['monitoring']['tensorboard_log_dir'] = os.path.join(
            base_path, f"{selected_model}_logs")

    # Handle MLflow tracking URI (HTTP URL - don't process as file path)
    # The mlflow_tracking_uri should remain as is if it's a URL
    if 'mlflow_tracking_uri' in config['monitoring']:
        # Only process if it's not a URL and not an absolute path
        uri = config['monitoring']['mlflow_tracking_uri']
        if not (uri.startswith('http://') or uri.startswith('https://') or os.path.isabs(uri)):
            rel_path = uri.lstrip('./')
            base_path = os.path.join(script_dir, rel_path)
            config['monitoring']['mlflow_tracking_uri'] = base_path

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

    # Use model-specific batch size if available, otherwise use global setting
    batch_size = model_config.get(
        'batch_size', config['dataset']['batch_size'])

    # Use model-specific image size if available, otherwise use global setting
    img_size = model_config.get('img_size', config['dataset']['img_size'])
    print(f"📐 Using image size: {img_size}x{img_size}")

    # Use pre-computed normalization statistics if available, otherwise compute them
    if USE_PRECOMPUTED_STATS:
        print("📊 Using pre-computed normalization statistics...")
        mean = torch.tensor(NORMALIZATION_MEAN)
        std = torch.tensor(NORMALIZATION_STD)
        print(f"📊 Pre-computed mean: {NORMALIZATION_MEAN}")
        print(f"📊 Pre-computed std: {NORMALIZATION_STD}")

        # Load datasets with pre-computed normalization
        # Create a temporary config with model-specific image size
        temp_config = config.copy()
        temp_config['dataset'] = config['dataset'].copy()
        temp_config['dataset']['img_size'] = img_size

        train_ds, val_ds, _ = load_datasets(
            temp_config, mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD, grayscale=grayscale)
    else:
        # Load datasets without normalization first to compute mean/std
        # Create a temporary config with model-specific image size
        temp_config = config.copy()
        temp_config['dataset'] = config['dataset'].copy()
        temp_config['dataset']['img_size'] = img_size

        train_ds_raw, val_ds_raw, _ = load_datasets(
            temp_config, mean=None, std=None, grayscale=grayscale)

        # Compute mean/std on training set only
        print("📊 Computing mean/std on training set...")
        mean, std = calculate_mean_std(train_ds_raw,
                                       batch_size=batch_size,
                                       num_workers=config['dataset']['num_workers'],
                                       pin_memory=config['dataset']['pin_memory'])

        print(f"📊 Training set mean: {mean.tolist()}")
        print(f"📊 Training set std: {std.tolist()}")

        # Reload datasets with proper normalization
        print("📊 Reloading datasets with normalization...")
        train_ds, val_ds, _ = load_datasets(
            temp_config, mean=mean.tolist(), std=std.tolist(), grayscale=grayscale)

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

    print(f"📊 Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Use Adamax for xception_medical (optimized for full fine-tuning), AdamW for others
    if selected_model == 'xception_medical':
        optimizer = optim.Adamax(classifier_params, lr=model_config['lr'])
        print(
            f"🔧 Using Adamax optimizer (optimized for full fine-tuning) with LR: {model_config['lr']}")
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

    # Add model info to TensorBoard
    writer.add_text('Model/Info', f'Model: {selected_model}', 0)
    writer.add_text('Model/Config', f'Model Config: {model_config}', 0)
    writer.add_text(
        'Dataset/Info', f'Dataset: {len(train_ds)} train, {len(val_ds)} val samples', 0)

    # Weights & Biases
    wandb_config = config['monitoring']['wandb']

    # Create a unique run name if not provided
    if not wandb_config.get('name'):
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{selected_model}-{timestamp}"
    else:
        # Add model name to existing name if not already present
        if selected_model not in wandb_config['name']:
            run_name = f"{selected_model}-{wandb_config['name']}"
        else:
            run_name = wandb_config['name']

    # Add model name to tags if not already present
    wandb_tags = wandb_config['tags'].copy() if wandb_config['tags'] else []
    if selected_model not in wandb_tags:
        wandb_tags.append(selected_model)

    # Create descriptive notes with model info
    model_notes = f"Model: {selected_model} | {wandb_config.get('notes', '')}"

    wandb.init(
        project=wandb_config['project'],
        entity=wandb_config['entity'],
        name=run_name,
        tags=wandb_tags,
        notes=model_notes,
        config={
            **model_config,
            **config['dataset'],
            **config['train'],
            'model_type': selected_model,
            'normalization_mean': mean.tolist(),
            'normalization_std': std.tolist()
        }
    )

    # MLflow (optional - skip if connection fails)
    mlflow_enabled = True
    mlflow_context = None

    try:
        # Setup MLflow with model-specific experiment
        mlflow.set_tracking_uri(config['monitoring']['mlflow_tracking_uri'])

        # Create a unique experiment name with timestamp to avoid conflicts
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_experiment_name = f"{config['monitoring']['mlflow_experiment_name']}-{selected_model}-{timestamp}"

        try:
            mlflow.set_experiment(model_experiment_name)
            print(f"📊 MLflow: {model_experiment_name}")
        except Exception as e:
            print(
                f"⚠️  Failed to set experiment '{model_experiment_name}': {e}")
            # Fallback to a simpler experiment name
            fallback_experiment_name = f"brain-cancer-mri-{selected_model}"
            print(
                f"🔄 Using fallback experiment name: {fallback_experiment_name}")
            mlflow.set_experiment(fallback_experiment_name)

        # Try to start MLflow run
        try:
            mlflow_context = mlflow.start_run(
                run_name=f"{selected_model}-training")
            mlflow_context.__enter__()
            print("✅ MLflow run started successfully")
        except Exception as e:
            print(f"⚠️  Failed to start MLflow run: {e}")
            print("🔄 Continuing without MLflow...")
            mlflow_enabled = False
            mlflow_context = None

    except Exception as e:
        print(f"⚠️  MLflow setup failed: {e}")
        print("🔄 Continuing without MLflow...")
        mlflow_enabled = False
        mlflow_context = None

        # Log MLflow parameters (only if MLflow is enabled)
    if mlflow_enabled and mlflow_context:
        try:
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
                "augmentation": config['transform']['augmentation'],
                "normalization_mean": mean.tolist(),
                "normalization_std": std.tolist()
            })

            # Set run description
            mlflow.set_tag("model", selected_model)
            mlflow.set_tag(
                "description", f"Brain Cancer MRI training with {selected_model}")
            print("✅ MLflow parameters logged successfully")
        except Exception as e:
            print(f"⚠️  Failed to log MLflow parameters: {e}")
            print("🔄 Continuing without MLflow logging...")
            mlflow_enabled = False

        # Print training configuration
        hw_info = get_hardware_info()

        print("="*60)
        print("🚀 Starting Brain Cancer MRI Training")
        print("="*60)
        print("🔍 About to start training loop...")
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
        print(
            f"🔄 Starting training loop for {config['train']['epochs']} epochs...")

        for epoch in range(config['train']['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            epoch_start_time = time.time()

            print(f"\n📚 Epoch {epoch+1}/{config['train']['epochs']}")
            print("-" * 50)
            print(f"  🔄 Starting training phase...")
            print(f"  📊 Total batches: {len(train_loader)}")

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

                # Also print at the end of each epoch
                if batch_idx == len(train_loader) - 1:
                    print(f"  ✅ Completed {len(train_loader)} batches")

            # Validation phase
            print(f"  🔍 Starting validation phase...")
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

                # Log best model to MLflow and Wandb
                try:
                    print(
                        f"🏆 New best model saved! Validation accuracy: {val_acc:.2f}%")
                    print(
                        f"💾 Best model checkpoint saved to: {best_model_path}")

                    # Log best model to MLflow
                    if mlflow_enabled and mlflow_context:
                        try:
                            # Log the model artifact (for backward compatibility)
                            mlflow.log_artifact(best_model_path, "best_model")

                            # Log the model as MLflow PyTorch model (enables UI registration)
                            mlflow.pytorch.log_model(
                                pytorch_model=model,
                                artifact_path="best_model",
                                code_paths=["models/", "data/", "utils/"]
                            )

                            mlflow.set_tag("best_model_path", best_model_path)
                            mlflow.set_tag("best_val_accuracy",
                                           f"{val_acc:.2f}%")
                            mlflow.set_tag("best_model_epoch", epoch)
                            mlflow.set_tag("model_registered", "true")
                            print(
                                "✅ Best model logged to MLflow (with registration capability)")
                        except Exception as e:
                            print(
                                f"⚠️  Failed to log best model to MLflow: {e}")

                    # Log best model to Wandb
                    try:
                        wandb.save(best_model_path)
                        wandb.run.summary["best_val_accuracy"] = val_acc
                        wandb.run.summary["best_model_epoch"] = epoch
                        wandb.run.summary["best_model_path"] = best_model_path
                        print("✅ Best model logged to Wandb")
                    except Exception as e:
                        print(f"⚠️  Failed to log best model to Wandb: {e}")

                except Exception as e:
                    print(f"⚠️  Failed to save best model: {e}")

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

            # Log to MLflow (only if enabled)
            if mlflow_enabled and mlflow_context:
                try:
                    mlflow.log_metrics({
                        'train_loss': train_loss_avg,
                        'val_loss': val_loss_avg,
                        'train_accuracy': train_acc,
                        'val_accuracy': val_acc,
                        'epoch_time': epoch_time
                    }, step=epoch)
                except Exception as e:
                    print(f"⚠️  MLflow logging failed: {e}")
                    print("🔄 Disabling MLflow logging for remaining epochs...")
                    mlflow_enabled = False

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
                'memory_percent': hw_metrics.get('memory_percent', 0),
                'normalization_mean_r': mean[0].item(),
                'normalization_mean_g': mean[1].item(),
                'normalization_mean_b': mean[2].item(),
                'normalization_std_r': std[0].item(),
                'normalization_std_g': std[1].item(),
                'normalization_std_b': std[2].item()
            }

            # Add GPU metrics if available
            if hw_metrics.get('gpu_available', False):
                wandb_metrics.update({
                    'gpu_utilization': hw_metrics.get('gpu_utilization', 0),
                    'gpu_memory_percent': hw_metrics.get('gpu_memory_percent', 0),
                    'gpu_temperature': hw_metrics.get('gpu_temperature', 0)
                })

            # Log to Wandb (outside MLflow context to avoid conflicts)
            try:
                wandb.log(wandb_metrics)
            except Exception as e:
                print(f"⚠️  Wandb logging error: {e}")

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

                # Log model to MLflow for registration capability
                if mlflow_enabled and mlflow_context:
                    try:
                        mlflow.pytorch.log_model(
                            pytorch_model=model,
                            artifact_path=f"model_epoch_{epoch+1}",
                            code_paths=["models/", "data/", "utils/"]
                        )
                        print(
                            f"✅ Model logged to MLflow for epoch {epoch+1} (can be registered from UI)")
                    except Exception as e:
                        print(
                            f"⚠️  Failed to log model to MLflow for epoch {epoch+1}: {e}")

        # Training completed - model registration handled separately
        print(
            f"🏆 Training completed! Best validation accuracy: {best_val_acc*100:.2f}%")
        if best_val_acc > 0.85:
            print(f"🎯 Model achieved excellent performance - ready for registration!")
        else:
            print(
                f"📊 Model achieved {best_val_acc*100:.2f}% validation accuracy - consider further tuning")

        print(f"💾 Best model saved to: {best_model_path}")
        print(
            f"📝 To register the model with MLflow, run: python register_model.py --model {selected_model}")

        # Final logging of best model to MLflow and Wandb
        try:
            # Log final best model to MLflow
            if mlflow_enabled and mlflow_context:
                try:
                    # Log the model artifact (for backward compatibility)
                    mlflow.log_artifact(best_model_path, "final_best_model")

                    # Log the model as MLflow PyTorch model (enables UI registration)
                    mlflow.pytorch.log_model(
                        pytorch_model=model,
                        artifact_path="model",
                        registered_model_name=f"brain-cancer-mri-{selected_model}",
                        # Include source code
                        code_paths=["models/", "data/", "utils/"]
                    )

                    mlflow.set_tag("final_best_val_accuracy",
                                   f"{best_val_acc*100:.2f}%")
                    mlflow.set_tag("final_best_model_epoch", "final")
                    mlflow.set_tag("model_registered", "true")
                    print(
                        "✅ Final best model logged to MLflow (with registration capability)")
                except Exception as e:
                    print(f"⚠️  Failed to log final best model to MLflow: {e}")

            # Log final best model to Wandb
            try:
                wandb.save(best_model_path)
                wandb.run.summary["final_best_val_accuracy"] = best_val_acc * 100
                wandb.run.summary["final_best_model_path"] = best_model_path
                print("✅ Final best model logged to Wandb")
            except Exception as e:
                print(f"⚠️  Failed to log final best model to Wandb: {e}")

        except Exception as e:
            print(f"⚠️  Failed to log final best model: {e}")

        # Log model to wandb
        wandb.save("model.pth")

        # Close monitoring
        writer.close()
        wandb.finish()

        # Close MLflow context if it was opened
        if mlflow_enabled and mlflow_context:
            mlflow_context.__exit__(None, None, None)

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
        print("🔍 Training loop completed!")


if __name__ == '__main__':
    print("🎯 Script started - Brain Cancer MRI Training")
    print("="*60)

    parser = argparse.ArgumentParser(
        description='Brain Cancer MRI Classification Training')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file (default: config/config.yaml)')
    parser.add_argument('--model', type=str, default=None,
                        help='Override model selection (resnet18, resnet50, swin_t, swin_s, efficientnet_b0, vit_b_16, medical_cnn, xception_medical)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--grayscale', action='store_true',
                        help='Use grayscale input (single channel) instead of RGB conversion')

    args = parser.parse_args()

    print(f"📋 Arguments parsed:")
    print(f"  Model: {args.model or 'default from config'}")
    print(f"  Config: {args.config}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Grayscale: {args.grayscale}")
    print("="*60)

    # Load config and apply overrides
    if args.model or args.epochs or args.batch_size or args.lr or args.grayscale:
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
        if args.grayscale:
            if 'dataset' not in config:
                config['dataset'] = {}
            config['dataset']['grayscale'] = True

        # Save temporary config
        temp_config_path = 'temp_config.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)

        main(temp_config_path, grayscale=args.grayscale)

        # Clean up
        try:
            os.remove(temp_config_path)
        except FileNotFoundError:
            pass  # File was already removed or doesn't exist
    else:
        main(args.config, grayscale=args.grayscale)
