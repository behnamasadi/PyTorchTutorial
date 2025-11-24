import argparse
import yaml
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from models.model import get_model
from data.dataset import load_datasets

# Monitoring imports
import mlflow
import wandb


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
    plt.title('Confusion Matrix - Brain Cancer MRI Classification')
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
        f.write(f"Brain Cancer MRI Model Evaluation Report\n")
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
        config_path = os.path.join(script_dir, config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve all paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Fix dataset path
    if not os.path.isabs(config['dataset']['path']):
        rel_path = config['dataset']['path'].lstrip('./')
        config['dataset']['path'] = os.path.join(script_dir, rel_path)

    # Get selected model configuration (allow override from command line)
    selected_model = model_override if model_override else config['model']

    # Fix output directory and make it model-specific
    if not os.path.isabs(config['train']['output_dir']):
        rel_path = config['train']['output_dir'].lstrip('./')
        base_output_dir = os.path.join(script_dir, rel_path)
    else:
        base_output_dir = config['train']['output_dir']

    # Create model-specific output directory
    model_output_dir = os.path.join(
        base_output_dir, f"{selected_model}_outputs")
    config['train']['output_dir'] = model_output_dir

    print("🧠 **Brain Cancer MRI Model Evaluation**")
    print("=" * 50)

    if selected_model not in config['models']:
        available_models = ', '.join(config['models'].keys())
        raise ValueError(
            f"Model '{selected_model}' not found in config. Available models: {available_models}")

    model_config = config['models'][selected_model]

    print(f"📋 Model: {model_config['name']}")
    print(f"📋 Classes: {model_config['num_classes']}")

    # Use model-specific image size if available
    if 'img_size' in model_config:
        config['dataset']['img_size'] = model_config['img_size']
        print(
            f"🖼️  Using model-specific image size: {model_config['img_size']}x{model_config['img_size']}")

    # Debug: Print configuration details
    print(f"🔍 Dataset path: {config['dataset']['path']}")
    print(f"🔍 Image size: {config['dataset']['img_size']}")
    print(f"🔍 Batch size: {config['dataset']['batch_size']}")
    print(f"🔍 Num workers: {config['dataset']['num_workers']}")

    # Load model architecture
    weights = eval(model_config['weights']
                   ) if model_config['weights'] else None
    model, classifier_params = get_model(
        model_config['name'],
        model_config['num_classes'],
        weights
    )

    # Load trained checkpoint - only look for the specific model
    checkpoint_paths = [
        # Primary: Early stopping checkpoint for the specific model
        os.path.join(config['train']['output_dir'], 'best_model.pth'),
        # Fallback: Regular checkpoint for the specific model
        os.path.join(config['train']['output_dir'], 'checkpoint_latest.pth'),
        # Note: Removed hardcoded MLflow path to prevent loading wrong model
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
                    best_model_path = os.path.join(item_path, 'best_model.pth')
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
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        # MLflow saves the entire model
        model = checkpoint

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"🖥️  Using device: {device}")

    # Import normalization constants
    try:
        from normalization_constants import NORMALIZATION_MEAN, NORMALIZATION_STD
        print(
            f"📊 Using pre-computed normalization: mean={NORMALIZATION_MEAN}, std={NORMALIZATION_STD}")
    except ImportError:
        print("⚠️  No pre-computed normalization found. Computing from training data...")
        # Fallback: compute normalization from training data
        from utils.helpers import calculate_mean_std
        train_ds, _, _ = load_datasets(
            config, mean=None, std=None, grayscale=False)
        NORMALIZATION_MEAN, NORMALIZATION_STD = calculate_mean_std(
            train_ds, config['dataset']['batch_size'])
        print(
            f"📊 Computed normalization: mean={NORMALIZATION_MEAN}, std={NORMALIZATION_STD}")

    # Load test dataset with proper normalization
    _, _, test_ds = load_datasets(
        config, mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD, grayscale=False)
    test_loader = DataLoader(test_ds, batch_size=config['dataset']['batch_size'],
                             shuffle=False, num_workers=config['dataset']['num_workers'])

    print(f"📊 Test set size: {len(test_ds)} samples")

    # Class names (adjust based on your dataset structure)
    class_names = ['glioma', 'meningioma',
                   'pituitary']  # 3 classes as per config

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
    test_experiment_name = f"brain-cancer-mri-test-{selected_model}"
    mlflow.set_experiment(test_experiment_name)

    # Simple wandb logging for test
    wandb.init(
        project="brain-cancer-mri-test",
        name=f"{selected_model}_simple_test",
        tags=["test", "simple", "medical-ai", selected_model],
        notes=f"Simple test evaluation for {selected_model}",
        config={
            "model_name": selected_model,
            "model_type": model_config['name'],
            "num_classes": model_config['num_classes'],
            "test_samples": len(y_true)
        }
    )

    # Log to MLflow and wandb
    with mlflow.start_run(run_name=f"{selected_model}_simple_test"):
        # Log basic metrics
        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "test_samples": len(y_true)
        })

        mlflow.log_params({
            "model_name": model_config['name'],
            "model_type": selected_model,
            "num_classes": model_config['num_classes']
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
    print(f"🔮 Wandb project: brain-cancer-mri-test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate trained Brain Cancer MRI model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                        help='Model to evaluate (overrides config file). Choose from: resnet18, swin_t, swin_s, efficientnet_b0, vit_b_16, medical_cnn, xception_medical')
    args = parser.parse_args()
    main(args.config, args.model)
