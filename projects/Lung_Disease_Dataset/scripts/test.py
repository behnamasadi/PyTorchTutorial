import mlflow
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import argparse
import yaml
import os
import sys
from pathlib import Path
import warnings
# Suppress pydantic warnings from wandb/mlflow dependencies
# These warnings come from pydantic's internal schema generation when processing Field() definitions
# They're harmless and come from dependencies (wandb/mlflow), not our code
# MUST be set before importing wandb/mlflow to be effective
warnings.filterwarnings("ignore", category=UserWarning,
                        module="pydantic._internal._generate_schema")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


# Add src directory to path for imports (MUST be before other imports)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from lung_disease_dataset.models.model import get_model  # noqa: E402

# Note: test.py uses ImageFolder directly, not load_datasets

# Monitoring imports


def evaluate_model(model, test_loader, device, class_names):
    """Comprehensive model evaluation on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print("🧪 Running inference on test set...")
    print(f"🔍 Model device: {next(model.parameters()).device}")
    print(f"🔍 Data device: {device}")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            # Debug: Check data statistics
            if batch_idx == 0:
                print(f"🔍 First batch - Data shape: {data.shape}")
                print(
                    f"🔍 First batch - Data range: [{data.min():.4f}, {data.max():.4f}]")
                print(f"🔍 First batch - Data mean: {data.mean():.4f}")
                print(f"🔍 First batch - Data std: {data.std():.4f}")
                print(
                    f"🔍 First batch - Target distribution: {torch.bincount(target)}")

            output = model(data)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(test_loader)}")

    # Debug: Check prediction distribution
    print(f"🔍 Prediction distribution: {np.bincount(all_preds)}")
    print(f"🔍 True label distribution: {np.bincount(all_labels)}")

    return np.array(all_preds), np.array(all_labels)


def generate_evaluation_report(y_true, y_pred, class_names, output_dir):
    """Generate comprehensive evaluation metrics and visualizations"""

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n📊 **Test Set Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)**")

    # Classification report
    print("\n📋 **Classification Report:**")
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Lung Disease Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"💾 Confusion matrix saved to: {cm_path}")
    plt.close()

    # Per-class analysis
    print("\n🔍 **Per-Class Analysis:**")
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == i).mean()
            print(
                f"  {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%) - {class_mask.sum()} samples")

    # Save detailed report
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Lung Disease Dataset Model Evaluation Report\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(
            f"Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\n\nConfusion Matrix:\n{cm}")

    print(f"📄 Detailed report saved to: {report_path}")

    return accuracy, report, cm


def main(config_path, model_override=None):
    # Resolve absolute path to config file
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        # Try relative to script dir first, then project root
        if os.path.exists(os.path.join(script_dir, config_path)):
            config_path = os.path.join(script_dir, config_path)
        elif os.path.exists(os.path.join(project_root, config_path)):
            config_path = os.path.join(project_root, config_path)
        else:
            config_path = os.path.join(script_dir, config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve all paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Handle different config structures - load model.yaml and data.yaml if needed
    if 'models' not in config:
        # Load model.yaml
        model_yaml_path = os.path.join(project_root, 'configs', 'model.yaml')
        if os.path.exists(model_yaml_path):
            with open(model_yaml_path, 'r') as f:
                model_config_data = yaml.safe_load(f)
                if 'models' in model_config_data:
                    config['models'] = model_config_data['models']

    # Handle dataset path - check multiple possible locations
    if 'dataset' in config and 'path' in config['dataset']:
        dataset_path = config['dataset']['path']
    elif 'data' in config and 'test_path' in config['data']:
        dataset_path = config['data']['test_path']
    elif 'paths' in config and 'test' in config['paths']:
        dataset_path = config['paths']['test']
    else:
        dataset_path = './data/test'

    if not os.path.isabs(dataset_path):
        rel_path = dataset_path.lstrip('./')
        dataset_path = os.path.join(project_root, rel_path)

    # Ensure dataset config exists
    if 'dataset' not in config:
        config['dataset'] = {}
    config['dataset']['path'] = dataset_path

    # Get selected model configuration (allow override from command line)
    selected_model = model_override if model_override else config.get(
        'model', None)
    if not selected_model:
        raise ValueError(
            "Model must be specified via --model argument or in config file")

    # Fix output directory - check multiple possible locations
    if 'train' in config and 'output_dir' in config['train']:
        output_dir = config['train']['output_dir']
    else:
        output_dir = './checkpoints'

    if not os.path.isabs(output_dir):
        rel_path = output_dir.lstrip('./')
        base_output_dir = os.path.join(project_root, rel_path)
    else:
        base_output_dir = output_dir

    # Ensure train config exists
    if 'train' not in config:
        config['train'] = {}
    config['train']['output_dir'] = base_output_dir

    # Create model-specific output directory
    model_output_dir = os.path.join(
        base_output_dir, f"{selected_model}_outputs")
    config['train']['output_dir'] = model_output_dir

    print("🫁 **Lung Disease Dataset Model Evaluation**")
    print("=" * 50)

    if selected_model not in config['models']:
        available_models = ', '.join(config['models'].keys())
        raise ValueError(
            f"Model '{selected_model}' not found in config. Available models: {available_models}")

    model_config = config['models'][selected_model]

    print(f"📋 Model: {model_config['name']}")
    print(f"📋 Classes: {model_config['num_classes']}")

    # Use model-specific image size if available
    img_size = model_config.get('input_size', config.get(
        'dataset', {}).get('image_size', 224))
    config['dataset']['img_size'] = img_size
    if 'input_size' in model_config:
        print(
            f"🖼️  Using model-specific image size: {model_config['input_size']}x{model_config['input_size']}")

    # Debug: Print configuration details
    print(f"🔍 Dataset path: {config['dataset']['path']}")
    print(f"🔍 Image size: {img_size}")
    batch_size = config.get('dataset', {}).get(
        'batch_size', model_config.get('batch_size', 32))
    num_workers = config.get('dataset', {}).get(
        'num_workers', config.get('dataloader', {}).get('num_workers', 4))
    print(f"🔍 Batch size: {batch_size}")
    print(f"🔍 Num workers: {num_workers}")

    # Load model architecture
    model = get_model(
        model_config['name'],
        model_config['num_classes'],
        pretrained=model_config.get('pretrained', True)
    )

    # Load trained checkpoint - check both model-specific output dir and root checkpoints dir
    base_checkpoints_dir = base_output_dir  # Root checkpoints directory

    checkpoint_paths = [
        # Primary: Model-specific best model checkpoint in output dir
        os.path.join(config['train']['output_dir'],
                     f'{selected_model}-training_best_model.pth'),
        # Also check root checkpoints directory
        os.path.join(base_checkpoints_dir,
                     f'{selected_model}-training_best_model.pth'),
        # Fallback: Model-specific last model checkpoint
        os.path.join(config['train']['output_dir'],
                     f'{selected_model}-training_last_model.pth'),
        os.path.join(base_checkpoints_dir,
                     f'{selected_model}-training_last_model.pth'),
        # Legacy: Old naming convention for backward compatibility
        os.path.join(config['train']['output_dir'], 'best_model.pth'),
        os.path.join(base_checkpoints_dir, 'best_model.pth'),
        os.path.join(config['train']['output_dir'], 'checkpoint_latest.pth'),
    ]

    print(
        f"🔍 Looking for {selected_model} checkpoint in: {config['train']['output_dir']}")
    print(f"🔍 Checkpoint paths to try: {checkpoint_paths}")

    checkpoint_path = None
    for path in checkpoint_paths:
        print(f"🔍 Checking: {path} - Exists: {os.path.exists(path)}")
        if os.path.exists(path):
            checkpoint_path = path
            break

    if checkpoint_path is None:
        # Check what models are actually available
        available_checkpoints = []
        base_dir = os.path.dirname(config['train']['output_dir'])
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path) and item.endswith('_outputs'):
                    model_name = item.replace('_outputs', '')
                    # Check for model-specific naming first, then legacy naming
                    best_model_path = os.path.join(
                        item_path, f'{model_name}-training_best_model.pth')
                    if not os.path.exists(best_model_path):
                        best_model_path = os.path.join(
                            item_path, 'best_model.pth')
                    if os.path.exists(best_model_path):
                        available_checkpoints.append(model_name)

        if available_checkpoints:
            raise FileNotFoundError(
                f"❌ No trained model found for '{selected_model}'.\n"
                f"📋 Available trained models: {', '.join(available_checkpoints)}\n"
                f"💡 To evaluate available models, use:\n"
                f"   python3 test.py --model {available_checkpoints[0]}\n"
                f"🚀 To train {selected_model}, use:\n"
                f"   python3 train.py --model {selected_model}")
        else:
            raise FileNotFoundError(
                f"No trained models found. Train a model first with: python3 train.py")

    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            # New format from train.py: {'stage': ..., 'epoch': ..., 'val_accuracy': ..., 'state_dict': ...}
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            if 'val_accuracy' in checkpoint:
                print(
                    f"📊 Checkpoint validation accuracy: {checkpoint['val_accuracy']*100:.2f}%")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            # Direct state dict
            model.load_state_dict(checkpoint, strict=False)
    else:
        # MLflow saves the entire model
        model = checkpoint

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"🖥️  Using device: {device}")

    # Import normalization constants
    from torchvision import datasets, transforms
    try:
        from lung_disease_dataset.utils.normalization_constants import NORMALIZATION_MEAN, NORMALIZATION_STD
        print(
            f"📊 Using pre-computed normalization: mean={NORMALIZATION_MEAN}, std={NORMALIZATION_STD}")
    except ImportError:
        print("⚠️  Using default ImageNet normalization...")
        NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
        NORMALIZATION_STD = [0.229, 0.224, 0.225]

    # Build transforms for test dataset
    img_size = config.get('dataset', {}).get(
        'img_size', model_config.get('input_size', 224))
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
    ])

    # Load test dataset
    test_path = config.get('dataset', {}).get('path', './data/test')
    if not os.path.isabs(test_path):
        test_path = os.path.join(script_dir, test_path)
    test_ds = datasets.ImageFolder(test_path, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    print(f"📊 Test set size: {len(test_ds)} samples")

    # Class names - get from config or dataset
    if 'dataset' in config and 'class_names' in config['dataset']:
        class_names = config['dataset']['class_names']
    elif 'class_names' in config:
        class_names = config['class_names']
    else:
        # Fallback: try to get from dataset folder structure
        test_path = config['dataset']['path']
        if os.path.exists(test_path):
            class_names = sorted([d for d in os.listdir(test_path)
                                 if os.path.isdir(os.path.join(test_path, d)) and not d.startswith('.')])
        else:
            # Default fallback
            class_names = [f'class_{i}' for i in range(
                model_config['num_classes'])]

    print(f"📋 Class names: {class_names}")

    # Run evaluation
    y_pred, y_true = evaluate_model(model, test_loader, device, class_names)

    # Generate comprehensive report
    os.makedirs(config['train']['output_dir'], exist_ok=True)
    accuracy, report, cm = generate_evaluation_report(
        y_true, y_pred, class_names, config['train']['output_dir'])

    # Setup logging for test results
    print("📊 Logging test results...")

    # MLflow setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mlflow_uri = os.path.join(script_dir, 'mlruns')
    mlflow.set_tracking_uri(mlflow_uri)

    # Create test-specific experiment
    test_experiment_name = f"lung-disease-dataset-test-{selected_model}"
    mlflow.set_experiment(test_experiment_name)

    # Simple wandb logging for test with model name from checkpoint
    # Extract model name from checkpoint path if available
    model_name_from_checkpoint = selected_model
    if checkpoint_path:
        checkpoint_filename = os.path.basename(checkpoint_path)
        # Try to extract model name from filename (e.g., "convnextv2_base-training_best_model.pth" -> "convnextv2_base")
        if '-training_' in checkpoint_filename:
            model_name_from_checkpoint = checkpoint_filename.split(
                '-training_')[0]

    wandb.init(
        project="lung-disease-dataset-test",
        name=f"{model_name_from_checkpoint}_simple_test",
        tags=["test", "simple", "medical-ai", model_name_from_checkpoint],
        notes=f"Simple test evaluation for {model_name_from_checkpoint}",
        config={
            "model_name": model_name_from_checkpoint,
            "model_type": model_config['name'],
            "num_classes": model_config['num_classes'],
            "test_samples": len(y_true),
            "checkpoint_path": checkpoint_path
        }
    )

    # Log to MLflow and wandb
    with mlflow.start_run(run_name=f"{model_name_from_checkpoint}_simple_test"):
        # Log basic metrics
        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "test_samples": len(y_true)
        })

        mlflow.log_params({
            "model_name": model_config['name'],
            "model_type": model_name_from_checkpoint,
            "num_classes": model_config['num_classes'],
            "checkpoint_path": checkpoint_path
        })

        # Log artifacts
        report_path = os.path.join(
            config['train']['output_dir'], 'evaluation_report.txt')
        cm_path = os.path.join(
            config['train']['output_dir'], 'confusion_matrix.png')

        if os.path.exists(report_path):
            mlflow.log_artifact(report_path, "test_reports")
        if os.path.exists(cm_path):
            mlflow.log_artifact(cm_path, "test_visualizations")
            wandb.log({"confusion_matrix": wandb.Image(cm_path)})

        # Log to wandb
        wandb.log({
            "test_accuracy": accuracy,
            "test_samples": len(y_true)
        })

    wandb.finish()

    print(f"\n✅ **Evaluation Complete!**")
    print(f"🎯 Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 MLflow experiment: {test_experiment_name}")
    print(f"🔮 Wandb project: lung-disease-dataset-test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate trained Lung Disease Dataset model')
    parser.add_argument('--config', type=str, default='../configs/eval.yaml',
                        help='Path to config file (default: ../configs/eval.yaml)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model to evaluate. If not specified, all available trained models will be processed. Choose from: convnextv2_tiny, convnextv2_base, tf_efficientnetv2_s, tf_efficientnetv2_m, tf_efficientnetv2_l, regnety_004, regnety_006')
    args = parser.parse_args()

    # If no model specified, process all available models
    if args.model is None:
        # Load config to find available models
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)

        # Resolve config path
        if not os.path.isabs(args.config):
            if os.path.exists(os.path.join(script_dir, args.config)):
                config_path = os.path.join(script_dir, args.config)
            elif os.path.exists(os.path.join(project_root, args.config)):
                config_path = os.path.join(project_root, args.config)
            else:
                config_path = os.path.join(script_dir, args.config)
        else:
            config_path = args.config

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Handle different config structures
        if 'models' not in config:
            model_yaml_path = os.path.join(
                project_root, 'configs', 'model.yaml')
            if os.path.exists(model_yaml_path):
                with open(model_yaml_path, 'r') as f:
                    model_config_data = yaml.safe_load(f)
                    if 'models' in model_config_data:
                        config['models'] = model_config_data['models']

        # Find available models
        if 'train' in config and 'output_dir' in config['train']:
            output_dir_base = config['train']['output_dir']
        else:
            output_dir_base = './checkpoints'

        if not os.path.isabs(output_dir_base):
            rel_path = output_dir_base.lstrip('./')
            base_output_dir = os.path.join(project_root, rel_path)
        else:
            base_output_dir = output_dir_base

        # Find all available trained models
        available_models = []
        if os.path.exists(base_output_dir):
            # Check for model-specific checkpoints in root directory
            for filename in os.listdir(base_output_dir):
                if filename.endswith('-training_best_model.pth') or filename.endswith('-training_last_model.pth'):
                    model_name = filename.split('-training_')[0]
                    if model_name in config.get('models', {}):
                        if model_name not in available_models:
                            available_models.append(model_name)

            # Check for model-specific output directories
            for item in os.listdir(base_output_dir):
                item_path = os.path.join(base_output_dir, item)
                if os.path.isdir(item_path) and item.endswith('_outputs'):
                    model_name = item.replace('_outputs', '')
                    best_model_path = os.path.join(
                        item_path, f'{model_name}-training_best_model.pth')
                    if not os.path.exists(best_model_path):
                        best_model_path = os.path.join(
                            item_path, 'best_model.pth')
                    if os.path.exists(best_model_path) and model_name in config.get('models', {}):
                        if model_name not in available_models:
                            available_models.append(model_name)

        if not available_models:
            print(
                "❌ No trained models found. Train models first with: python train.py --model <model_name>")
            sys.exit(1)

        print(
            f"🫁 Processing {len(available_models)} trained model(s): {', '.join(available_models)}\n")

        # Process each model
        all_results = []
        for model_name in available_models:
            try:
                print(f"\n{'='*60}")
                print(f"Processing: {model_name}")
                print(f"{'='*60}\n")
                main(args.config, model_name)
                all_results.append(model_name)
            except Exception as e:
                print(f"❌ Error processing {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'='*60}")
        print(
            f"✅ Completed processing {len(all_results)} model(s): {', '.join(all_results)}")
        print(f"{'='*60}")
    else:
        main(args.config, args.model)
