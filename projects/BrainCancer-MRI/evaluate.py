#!/usr/bin/env python3
"""
Brain Cancer MRI Model Evaluation Script

This script provides comprehensive model evaluation capabilities including:
- Test set evaluation with detailed metrics
- Confusion matrix visualization
- Per-class performance analysis
- Model comparison utilities
- Robustness testing
- Medical AI specific validation metrics

Usage:
    python evaluate.py --model resnet18
    python evaluate.py --model efficientnet_b0 --config config/config.yaml
    python evaluate.py --model xception_medical --detailed
"""

import argparse
import yaml
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from models.model import get_model
from data.dataset import load_datasets
import json
from datetime import datetime

# Monitoring imports
import mlflow
import mlflow.pytorch
import wandb


def evaluate_model_on_test_set(model, test_loader, device, class_names, detailed=False):
    """
    Comprehensive model evaluation on test set

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test set
        device: Device to run evaluation on
        class_names: List of class names
        detailed: Whether to return detailed per-sample predictions

    Returns:
        dict: Evaluation metrics and predictions
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probabilities = []

    print("🧪 Running inference on test set...")

    inference_times = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            batch_start_time = time.time()

            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)

            batch_time = time.time() - batch_start_time
            inference_times.append(batch_time)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            if batch_idx % 10 == 0:
                print(f"  Processed batch {batch_idx+1}/{len(test_loader)} "
                      f"({len(data)} samples, {batch_time:.3f}s)")

    results = {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probabilities),
        'inference_time_per_batch': np.mean(inference_times),
        'inference_time_per_sample': np.mean(inference_times) / test_loader.batch_size,
        'total_samples': len(all_preds)
    }

    return results


def calculate_detailed_metrics(y_true, y_pred, y_proba, class_names):
    """
    Calculate comprehensive evaluation metrics

    Returns:
        dict: Detailed metrics including medical AI specific measures
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )

    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )

    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Medical AI specific metrics
    metrics = {
        'overall': {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        },
        'per_class': {},
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }

    # Per-class detailed analysis
    for i, class_name in enumerate(class_names):
        metrics['per_class'][class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': int(support[i]),
            'true_positives': int(cm[i, i]),
            'false_positives': int(cm[:, i].sum() - cm[i, i]),
            'false_negatives': int(cm[i, :].sum() - cm[i, i]),
            'specificity': int(cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]) /
            int(cm.sum() - cm[i, :].sum()) if (cm.sum() -
                                               cm[i, :].sum()) > 0 else 0.0
        }

    # Medical AI critical metrics
    # Sensitivity (recall) for each class - critical for medical diagnosis
    for i, class_name in enumerate(class_names):
        sensitivity = recall[i]
        if 'tumor' in class_name.lower():
            # For tumor classes, high sensitivity is critical (don't miss tumors)
            if sensitivity < 0.8:
                print(
                    f"⚠️  WARNING: Low sensitivity for {class_name}: {sensitivity:.3f}")
                print(f"   This could lead to missed tumor diagnoses!")

    return metrics


def plot_confusion_matrix(cm, class_names, output_dir, model_name):
    """Create and save confusion matrix visualization"""
    plt.figure(figsize=(10, 8))

    # Calculate percentages for annotation
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations with both count and percentage
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = cm_percent[i, j]
            row.append(f'{count}\n({percent:.1f}%)')
        annotations.append(row)

    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})

    plt.title(f'Confusion Matrix - {model_name}\nBrain Cancer MRI Classification',
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    # Save confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"💾 Confusion matrix saved to: {cm_path}")
    plt.close()

    return cm_path


def plot_per_class_metrics(metrics, output_dir, model_name):
    """Create per-class performance visualization"""
    classes = list(metrics['per_class'].keys())
    precision_scores = [metrics['per_class'][cls]['precision']
                        for cls in classes]
    recall_scores = [metrics['per_class'][cls]['recall'] for cls in classes]
    f1_scores = [metrics['per_class'][cls]['f1_score'] for cls in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, precision_scores, width,
                   label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title(f'Per-Class Performance Metrics - {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    metrics_path = os.path.join(output_dir, 'per_class_metrics.png')
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    print(f"📊 Per-class metrics plot saved to: {metrics_path}")
    plt.close()

    return metrics_path


def generate_evaluation_report(metrics, model_config, output_dir, model_name, inference_stats):
    """Generate comprehensive evaluation report"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Text report
    report_lines = [
        "=" * 80,
        f"Brain Cancer MRI Model Evaluation Report - {model_name}",
        "=" * 80,
        f"Generated: {timestamp}",
        f"Model: {model_config['name']}",
        f"Classes: {metrics['class_names']}",
        "",
        "🎯 OVERALL PERFORMANCE",
        "-" * 40,
        f"Test Accuracy: {metrics['overall']['accuracy']:.4f} ({metrics['overall']['accuracy']*100:.2f}%)",
        f"Macro Precision: {metrics['overall']['precision_macro']:.4f}",
        f"Macro Recall: {metrics['overall']['recall_macro']:.4f}",
        f"Macro F1-Score: {metrics['overall']['f1_macro']:.4f}",
        f"Weighted Precision: {metrics['overall']['precision_weighted']:.4f}",
        f"Weighted Recall: {metrics['overall']['recall_weighted']:.4f}",
        f"Weighted F1-Score: {metrics['overall']['f1_weighted']:.4f}",
        "",
        "⚡ INFERENCE PERFORMANCE",
        "-" * 40,
        f"Average inference time per sample: {inference_stats['inference_time_per_sample']*1000:.2f}ms",
        f"Average inference time per batch: {inference_stats['inference_time_per_batch']:.3f}s",
        f"Total samples evaluated: {inference_stats['total_samples']}",
        "",
        "🔍 PER-CLASS ANALYSIS",
        "-" * 40
    ]

    # Per-class detailed analysis
    for class_name, class_metrics in metrics['per_class'].items():
        report_lines.extend([
            f"",
            f"📋 {class_name.upper()}:",
            f"  Precision: {class_metrics['precision']:.4f}",
            f"  Recall (Sensitivity): {class_metrics['recall']:.4f}",
            f"  F1-Score: {class_metrics['f1_score']:.4f}",
            f"  Specificity: {class_metrics['specificity']:.4f}",
            f"  Support: {class_metrics['support']} samples",
            f"  True Positives: {class_metrics['true_positives']}",
            f"  False Positives: {class_metrics['false_positives']}",
            f"  False Negatives: {class_metrics['false_negatives']}"
        ])

    # Medical AI specific warnings
    report_lines.extend([
        "",
        "🏥 MEDICAL AI VALIDATION",
        "-" * 40
    ])

    # Check for medical AI quality thresholds
    overall_acc = metrics['overall']['accuracy']
    if overall_acc >= 0.95:
        report_lines.append(
            "✅ Excellent accuracy (≥95%) - Suitable for clinical decision support")
    elif overall_acc >= 0.90:
        report_lines.append(
            "✅ Very good accuracy (≥90%) - Good for medical screening")
    elif overall_acc >= 0.85:
        report_lines.append(
            "⚠️  Good accuracy (≥85%) - May need improvement for critical diagnoses")
    else:
        report_lines.append(
            "❌ Below medical AI threshold (<85%) - Requires improvement")

    # Check sensitivity for tumor classes
    for class_name, class_metrics in metrics['per_class'].items():
        sensitivity = class_metrics['recall']
        if 'tumor' in class_name.lower() or 'glioma' in class_name.lower() or 'meningioma' in class_name.lower():
            if sensitivity >= 0.90:
                report_lines.append(
                    f"✅ {class_name}: Excellent sensitivity ({sensitivity:.3f})")
            elif sensitivity >= 0.80:
                report_lines.append(
                    f"⚠️  {class_name}: Acceptable sensitivity ({sensitivity:.3f})")
            else:
                report_lines.append(
                    f"❌ {class_name}: Low sensitivity ({sensitivity:.3f}) - Risk of missed diagnoses")

    # Performance analysis
    if inference_stats['inference_time_per_sample'] <= 0.5:
        report_lines.append(
            "✅ Real-time inference capability (≤500ms per sample)")
    else:
        report_lines.append(
            f"⚠️  Slow inference ({inference_stats['inference_time_per_sample']*1000:.0f}ms per sample)")

    # Confusion matrix analysis
    report_lines.extend([
        "",
        "📊 CONFUSION MATRIX",
        "-" * 40
    ])

    cm = np.array(metrics['confusion_matrix'])
    for i, true_class in enumerate(metrics['class_names']):
        for j, pred_class in enumerate(metrics['class_names']):
            if i != j and cm[i, j] > 0:  # Misclassifications
                total_true = cm[i, :].sum()
                error_rate = cm[i, j] / total_true * 100
                if error_rate > 10:  # More than 10% error rate
                    report_lines.append(
                        f"⚠️  {true_class} → {pred_class}: {cm[i, j]} samples ({error_rate:.1f}%)")

    # Save text report
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"📄 Detailed evaluation report saved to: {report_path}")

    # Save JSON metrics for programmatic access
    json_metrics = {
        'timestamp': timestamp,
        'model_name': model_name,
        'metrics': metrics,
        'inference_stats': inference_stats
    }

    json_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(json_metrics, f, indent=2, default=str)

    print(f"📊 JSON metrics saved to: {json_path}")

    return report_path, json_path


def medical_ai_validation_checks(metrics, class_names):
    """
    Perform medical AI specific validation checks

    Returns:
        dict: Validation results and recommendations
    """
    validation_results = {
        'passed_medical_threshold': False,
        'sensitivity_warnings': [],
        'recommendations': [],
        'deployment_ready': False
    }

    overall_acc = metrics['overall']['accuracy']

    # Medical AI accuracy threshold (typically 85%+ for medical screening)
    if overall_acc >= 0.85:
        validation_results['passed_medical_threshold'] = True
    else:
        validation_results['recommendations'].append(
            f"Improve overall accuracy from {overall_acc:.3f} to ≥0.85 for medical deployment"
        )

    # Check sensitivity for each class (especially tumor classes)
    tumor_classes = [name for name in class_names if 'tumor' in name.lower() or
                     'glioma' in name.lower() or 'meningioma' in name.lower()]

    for class_name, class_metrics in metrics['per_class'].items():
        sensitivity = class_metrics['recall']

        if class_name in tumor_classes:
            if sensitivity < 0.80:
                validation_results['sensitivity_warnings'].append(
                    f"{class_name}: {sensitivity:.3f} (target: ≥0.80)"
                )
                validation_results['recommendations'].append(
                    f"Improve {class_name} sensitivity to reduce missed diagnoses"
                )

    # Check for class imbalance issues
    supports = [metrics['per_class'][cls]['support'] for cls in class_names]
    min_support = min(supports)
    max_support = max(supports)
    imbalance_ratio = max_support / \
        min_support if min_support > 0 else float('inf')

    if imbalance_ratio > 5:
        validation_results['recommendations'].append(
            f"Address class imbalance (ratio: {imbalance_ratio:.1f}:1) with data augmentation or class weighting"
        )

    # Overall deployment readiness
    validation_results['deployment_ready'] = (
        validation_results['passed_medical_threshold'] and
        len(validation_results['sensitivity_warnings']) == 0
    )

    return validation_results


def compare_models(model_results_dir):
    """
    Compare multiple trained models if available

    Args:
        model_results_dir: Base directory containing model-specific output folders
    """
    print("\n🔍 Searching for trained models to compare...")

    model_comparisons = []

    if not os.path.exists(model_results_dir):
        print(f"❌ Results directory not found: {model_results_dir}")
        return

    # Find all model output directories
    for item in os.listdir(model_results_dir):
        if item.endswith('_outputs'):
            model_name = item.replace('_outputs', '')
            model_path = os.path.join(model_results_dir, item)
            json_path = os.path.join(model_path, 'evaluation_metrics.json')

            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        model_data = json.load(f)
                    model_comparisons.append({
                        'name': model_name,
                        'accuracy': model_data['metrics']['overall']['accuracy'],
                        'f1_macro': model_data['metrics']['overall']['f1_macro'],
                        'inference_time': model_data['inference_stats']['inference_time_per_sample'],
                        'data': model_data
                    })
                except Exception as e:
                    print(f"⚠️  Could not load metrics for {model_name}: {e}")

    if len(model_comparisons) < 2:
        print("📝 Not enough models for comparison (need at least 2)")
        return

    # Sort by accuracy
    model_comparisons.sort(key=lambda x: x['accuracy'], reverse=True)

    print(f"\n📊 MODEL COMPARISON ({len(model_comparisons)} models)")
    print("=" * 80)
    print(f"{'Model':<20} {'Accuracy':<12} {'F1-Macro':<12} {'Inference (ms)':<15} {'Status'}")
    print("-" * 80)

    for model in model_comparisons:
        accuracy_pct = model['accuracy'] * 100
        f1_pct = model['f1_macro'] * 100
        inference_ms = model['inference_time'] * 1000

        # Status based on medical AI thresholds
        if model['accuracy'] >= 0.90:
            status = "🏆 Excellent"
        elif model['accuracy'] >= 0.85:
            status = "✅ Good"
        else:
            status = "⚠️  Needs improvement"

        print(
            f"{model['name']:<20} {accuracy_pct:>8.2f}%   {f1_pct:>8.2f}%   {inference_ms:>10.1f}ms    {status}")

    # Best model summary
    best_model = model_comparisons[0]
    print(f"\n🏆 BEST MODEL: {best_model['name']}")
    print(f"   Accuracy: {best_model['accuracy']*100:.2f}%")
    print(f"   F1-Score: {best_model['f1_macro']*100:.2f}%")
    print(
        f"   Inference: {best_model['inference_time']*1000:.1f}ms per sample")

    return model_comparisons


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Brain Cancer MRI Model Evaluation'
    )
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to evaluate (resnet18, swin_t, efficientnet_b0, etc.)')
    parser.add_argument('--detailed', action='store_true',
                        help='Generate detailed analysis and visualizations')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with other trained models')
    parser.add_argument('--medical-validation', action='store_true',
                        help='Perform medical AI specific validation checks')

    args = parser.parse_args()

    # Resolve config path
    if not os.path.isabs(args.config):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Fix dataset path
    if not os.path.isabs(config['dataset']['path']):
        rel_path = config['dataset']['path'].lstrip('./')
        config['dataset']['path'] = os.path.join(script_dir, rel_path)

    # Get model configuration
    if args.model not in config['models']:
        available_models = ', '.join(config['models'].keys())
        raise ValueError(
            f"Model '{args.model}' not found in config. Available models: {available_models}")

    model_config = config['models'][args.model]

    # Set up output directory
    if not os.path.isabs(config['train']['output_dir']):
        rel_path = config['train']['output_dir'].lstrip('./')
        base_output_dir = os.path.join(script_dir, rel_path)
    else:
        base_output_dir = config['train']['output_dir']

    output_dir = os.path.join(base_output_dir, f"{args.model}_outputs")

    print("🧠 Brain Cancer MRI Model Evaluation")
    print("=" * 50)
    print(f"📋 Model: {args.model} ({model_config['name']})")
    print(f"📂 Output directory: {output_dir}")

    # Check if trained model exists
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    if not os.path.exists(best_model_path):
        print(f"❌ No trained model found at: {best_model_path}")
        print(f"🚀 Train the model first: python train.py --model {args.model}")
        return

    # Load model architecture
    weights = eval(model_config['weights']
                   ) if model_config['weights'] else None
    model, classifier_params = get_model(
        model_config['name'],
        model_config['num_classes'],
        weights
    )

    # Load trained checkpoint
    print(f"📂 Loading checkpoint: {best_model_path}")
    checkpoint = torch.load(
        best_model_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'val_acc' in checkpoint:
            print(
                f"📊 Checkpoint validation accuracy: {checkpoint['val_acc']*100:.2f}%")
    else:
        model.load_state_dict(checkpoint)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"🖥️  Using device: {device}")

    # Use model-specific image size if available
    if 'img_size' in model_config:
        config['dataset']['img_size'] = model_config['img_size']
        print(
            f"🖼️  Using model-specific image size: {model_config['img_size']}x{model_config['img_size']}")

    # Load test dataset
    _, _, test_ds = load_datasets(config, mean=None, std=None)

    # Use model-specific batch size for evaluation
    eval_batch_size = model_config.get(
        'batch_size', config['dataset']['batch_size'])
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory']
    )

    print(f"📊 Test set: {len(test_ds)} samples")
    print(f"📦 Evaluation batch size: {eval_batch_size}")

    # Class names mapping (match the actual dataset structure)
    class_names = ['glioma', 'meningioma', 'pituitary']

    # Run evaluation
    print("\n🚀 Starting evaluation...")
    eval_start_time = time.time()

    eval_results = evaluate_model_on_test_set(
        model, test_loader, device, class_names, detailed=args.detailed
    )

    eval_time = time.time() - eval_start_time
    print(f"⏱️  Total evaluation time: {eval_time:.2f}s")

    # Calculate comprehensive metrics
    print("\n📊 Calculating metrics...")
    metrics = calculate_detailed_metrics(
        eval_results['labels'],
        eval_results['predictions'],
        eval_results['probabilities'],
        class_names
    )

    # Print summary results
    print(f"\n✅ **EVALUATION RESULTS**")
    print(
        f"🎯 Test Accuracy: {metrics['overall']['accuracy']:.4f} ({metrics['overall']['accuracy']*100:.2f}%)")
    print(f"📊 Macro F1-Score: {metrics['overall']['f1_macro']:.4f}")
    print(
        f"⚡ Avg inference time: {eval_results['inference_time_per_sample']*1000:.2f}ms per sample")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate visualizations
    if args.detailed:
        print("\n🎨 Generating visualizations...")

        # Confusion matrix
        plot_confusion_matrix(
            np.array(metrics['confusion_matrix']),
            class_names,
            output_dir,
            args.model
        )

        # Per-class metrics
        plot_per_class_metrics(metrics, output_dir, args.model)

    # Setup monitoring for evaluation logging
    print("\n📊 Setting up evaluation logging...")

    # MLflow setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mlflow_uri = os.path.join(script_dir, 'mlruns')
    mlflow.set_tracking_uri(mlflow_uri)

    # Create evaluation-specific experiment
    eval_experiment_name = f"brain-cancer-mri-evaluation-{args.model}"
    mlflow.set_experiment(eval_experiment_name)

    # Weights & Biases setup for evaluation
    wandb.init(
        project="brain-cancer-mri-evaluation",
        name=f"{args.model}_test_evaluation",
        tags=["evaluation", "test", "medical-ai", args.model],
        notes=f"Test set evaluation for {args.model} model",
        config={
            "model_name": args.model,
            "model_type": model_config['name'],
            "num_classes": model_config['num_classes'],
            "test_samples": eval_results['total_samples'],
            "evaluation_type": "comprehensive" if args.detailed else "basic"
        }
    )

    # Start MLflow run for evaluation
    with mlflow.start_run(run_name=f"{args.model}_test_evaluation"):

        # Log model configuration
        mlflow.log_params({
            "model_name": model_config['name'],
            "model_type": args.model,
            "num_classes": model_config['num_classes'],
            "test_samples": eval_results['total_samples'],
            "evaluation_type": "comprehensive" if args.detailed else "basic",
            "checkpoint_path": best_model_path
        })

        # Log comprehensive test metrics
        mlflow.log_metrics({
            "test_accuracy": metrics['overall']['accuracy'],
            "test_precision_macro": metrics['overall']['precision_macro'],
            "test_recall_macro": metrics['overall']['recall_macro'],
            "test_f1_macro": metrics['overall']['f1_macro'],
            "test_precision_weighted": metrics['overall']['precision_weighted'],
            "test_recall_weighted": metrics['overall']['recall_weighted'],
            "test_f1_weighted": metrics['overall']['f1_weighted'],
            "inference_time_per_sample_ms": eval_results['inference_time_per_sample'] * 1000,
            "inference_time_per_batch_s": eval_results['inference_time_per_batch'],
            "total_evaluation_time_s": eval_time
        })

        # Log per-class metrics
        for class_name, class_metrics in metrics['per_class'].items():
            mlflow.log_metrics({
                f"test_{class_name}_precision": class_metrics['precision'],
                f"test_{class_name}_recall": class_metrics['recall'],
                f"test_{class_name}_f1": class_metrics['f1_score'],
                f"test_{class_name}_specificity": class_metrics['specificity'],
                f"test_{class_name}_support": class_metrics['support']
            })

        # Log to Weights & Biases
        wandb_metrics = {
            "test_accuracy": metrics['overall']['accuracy'],
            "test_precision_macro": metrics['overall']['precision_macro'],
            "test_recall_macro": metrics['overall']['recall_macro'],
            "test_f1_macro": metrics['overall']['f1_macro'],
            "inference_time_ms": eval_results['inference_time_per_sample'] * 1000,
            "total_samples": eval_results['total_samples'],
            "evaluation_time_s": eval_time
        }

        # Add per-class metrics to wandb
        for class_name, class_metrics in metrics['per_class'].items():
            wandb_metrics.update({
                f"{class_name}_precision": class_metrics['precision'],
                f"{class_name}_recall": class_metrics['recall'],
                f"{class_name}_f1": class_metrics['f1_score'],
                f"{class_name}_specificity": class_metrics['specificity']
            })

        wandb.log(wandb_metrics)

        # Generate comprehensive report
        report_path, json_path = generate_evaluation_report(
            metrics, model_config, output_dir, args.model, eval_results
        )

    # Medical AI validation
    if args.medical_validation:
        print("\n🏥 Performing medical AI validation...")
        validation_results = medical_ai_validation_checks(metrics, class_names)

        print(f"\n📋 MEDICAL AI VALIDATION RESULTS:")
        print(
            f"   Accuracy threshold (≥85%): {'✅ PASSED' if validation_results['passed_medical_threshold'] else '❌ FAILED'}")
        print(
            f"   Deployment ready: {'✅ YES' if validation_results['deployment_ready'] else '❌ NO'}")

        if validation_results['sensitivity_warnings']:
            print(f"   ⚠️  Sensitivity warnings:")
            for warning in validation_results['sensitivity_warnings']:
                print(f"      - {warning}")

        if validation_results['recommendations']:
            print(f"   💡 Recommendations:")
            for rec in validation_results['recommendations']:
                print(f"      - {rec}")

        # Save validation results
        validation_path = os.path.join(output_dir, 'medical_validation.json')
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        print(f"🏥 Medical validation results saved to: {validation_path}")

        # Log medical validation to MLflow
        mlflow.log_metrics({
            "medical_threshold_passed": 1.0 if validation_results['passed_medical_threshold'] else 0.0,
            "deployment_ready": 1.0 if validation_results['deployment_ready'] else 0.0,
            "sensitivity_warnings_count": len(validation_results['sensitivity_warnings']),
            "recommendations_count": len(validation_results['recommendations'])
        })

        # Log medical validation to wandb
        wandb.log({
            "medical_validation": {
                "threshold_passed": validation_results['passed_medical_threshold'],
                "deployment_ready": validation_results['deployment_ready'],
                "warnings_count": len(validation_results['sensitivity_warnings']),
                "recommendations_count": len(validation_results['recommendations'])
            }
        })

        # Log artifacts to MLflow
        mlflow.log_artifact(report_path, "evaluation_reports")
        mlflow.log_artifact(json_path, "evaluation_reports")

        if args.detailed:
            # Log visualizations
            cm_path = os.path.join(output_dir, 'confusion_matrix.png')
            metrics_plot_path = os.path.join(
                output_dir, 'per_class_metrics.png')
            if os.path.exists(cm_path):
                mlflow.log_artifact(cm_path, "visualizations")
                # Log confusion matrix to wandb
                wandb.log({"confusion_matrix": wandb.Image(cm_path)})
            if os.path.exists(metrics_plot_path):
                mlflow.log_artifact(metrics_plot_path, "visualizations")
                # Log metrics plot to wandb
                wandb.log({"per_class_metrics": wandb.Image(metrics_plot_path)})

        if args.medical_validation:
            validation_path = os.path.join(
                output_dir, 'medical_validation.json')
            if os.path.exists(validation_path):
                mlflow.log_artifact(validation_path, "medical_validation")

    # Model comparison
    if args.compare:
        print("\n🔄 Comparing with other trained models...")
        comparison_results = compare_models(os.path.dirname(output_dir))

        # Log comparison results if available
        if comparison_results:
            # Log comparison summary to wandb
            comparison_table_data = []
            for model in comparison_results:
                comparison_table_data.append([
                    model['name'],
                    f"{model['accuracy']*100:.2f}%",
                    f"{model['f1_macro']*100:.2f}%",
                    f"{model['inference_time']*1000:.1f}ms"
                ])

            comparison_table = wandb.Table(
                data=comparison_table_data,
                columns=["Model", "Accuracy", "F1-Score", "Inference Time"]
            )
            wandb.log({"model_comparison": comparison_table})

    # Close monitoring
    wandb.finish()

    print(f"\n🎉 **Evaluation Complete!**")
    print(f"📁 All results saved to: {output_dir}")
    print(f"📄 View detailed report: {report_path}")
    print(f"📊 MLflow experiment: {eval_experiment_name}")
    print(f"🔮 Wandb project: brain-cancer-mri-evaluation")

    # Final recommendations
    if metrics['overall']['accuracy'] >= 0.90:
        print("🏆 Model shows excellent performance for medical AI!")
    elif metrics['overall']['accuracy'] >= 0.85:
        print("✅ Model meets medical AI standards.")
    else:
        print("⚠️  Model may need improvement before clinical deployment.")
        print(
            "💡 Consider: more training data, data augmentation, or different architecture.")


if __name__ == '__main__':
    main()
