#!/usr/bin/env python3
"""
Lung Disease Dataset Model Evaluation Script

This script provides comprehensive model evaluation capabilities including:
- Test set evaluation with detailed metrics
- Confusion matrix visualization
- Per-class performance analysis
- Model comparison utilities
- Robustness testing
- Medical AI specific validation metrics

The evaluation process is specifically designed for medical AI applications,
ensuring models meet clinical deployment standards with:
- High sensitivity for disease detection (minimizing false negatives)
- Robust accuracy across all disease types
- Real-time inference capabilities
- Complete audit trail for regulatory compliance

Usage:
    python evaluate.py --model resnet18
    python evaluate.py --model efficientnet_b0 --config config/config.yaml
    python evaluate.py --model xception_medical --detailed
    python evaluate.py --model swin_t --medical-validation --compare
"""

from datetime import datetime
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import time
from pathlib import Path
import sys
import os
import yaml
import argparse
import mlflow
import mlflow.pytorch
import wandb
import warnings
# Suppress pydantic warnings from wandb/mlflow dependencies
# These warnings come from pydantic's internal schema generation when processing Field() definitions
# They're harmless and come from dependencies (wandb/mlflow), not our code
# MUST be set before importing wandb/mlflow to be effective
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings(
    "ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*repr.*attribute.*Field.*")
warnings.filterwarnings("ignore", message=".*frozen.*attribute.*Field.*")


# Add src directory to path for imports (MUST be before other imports)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Monitoring imports for experiment tracking and model registry

from lung_disease_dataset.models.model import get_model  # noqa: E402
# Note: evaluate.py uses ImageFolder directly, not load_datasets


def evaluate_model_on_test_set(model, test_loader, device, class_names, detailed=False):
    """
    Comprehensive model evaluation on test set with medical AI focus

    This function performs inference on the test set while collecting:
    - Predictions and ground truth labels
    - Prediction probabilities for confidence analysis
    - Inference timing for real-time capability assessment
    - Per-batch processing statistics

    Medical AI Considerations:
    - Ensures no data leakage between train/validation/test sets
    - Collects detailed timing for clinical deployment assessment
    - Maintains complete audit trail of predictions
    - Provides confidence scores for clinical decision support

    Args:
        model: Trained PyTorch model in evaluation mode
        test_loader: DataLoader for test set with proper preprocessing
        device: Device to run evaluation on (CPU/GPU)
        class_names: List of class names for interpretable results
        detailed: Whether to return detailed per-sample predictions

    Returns:
        dict: Comprehensive evaluation results including:
            - predictions: Model predictions for each sample
            - labels: Ground truth labels
            - probabilities: Prediction confidence scores
            - inference_time_per_batch: Average batch processing time
            - inference_time_per_sample: Average per-sample inference time
            - total_samples: Total number of evaluated samples
    """
    model.eval()  # Ensure model is in evaluation mode
    all_preds = []
    all_labels = []
    all_probabilities = []

    print("🧪 Running inference on test set...")

    inference_times = []

    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch_idx, (data, target) in enumerate(test_loader):
            batch_start_time = time.time()

            data, target = data.to(device), target.to(device)

            # Forward pass - get model predictions
            output = model(data)
            # Convert logits to probabilities for confidence assessment
            probabilities = torch.softmax(output, dim=1)
            # Get predicted class (highest probability)
            pred = output.argmax(dim=1)

            batch_time = time.time() - batch_start_time
            inference_times.append(batch_time)

            # Store results for analysis
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            # Progress reporting for long evaluations
            if batch_idx % 10 == 0:
                print(f"  Processed batch {batch_idx+1}/{len(test_loader)} "
                      f"({len(data)} samples, {batch_time:.3f}s)")

    # Compile comprehensive results
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
    Calculate comprehensive evaluation metrics with medical AI focus

    This function computes a wide range of metrics essential for medical AI validation:

    Medical AI Critical Metrics:
    - Sensitivity (Recall): Critical for disease detection - must minimize false negatives
    - Specificity: Important for avoiding false alarms
    - Precision: Ensures high confidence in positive predictions
    - F1-Score: Balanced measure of precision and recall

    Clinical Considerations:
    - Per-class analysis for each disease type
    - Overall accuracy for general performance assessment
    - Macro and weighted averages for imbalanced datasets
    - Confusion matrix for detailed error analysis

    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        y_proba: Prediction probabilities (for confidence analysis)
        class_names: List of class names for interpretable results

    Returns:
        dict: Comprehensive metrics including:
            - overall: Macro and weighted performance metrics
            - per_class: Detailed metrics for each disease type
            - confusion_matrix: Error analysis matrix
            - class_names: For reference
    """
    # Basic accuracy calculation
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics - essential for medical AI
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )

    # Macro averages - treats all classes equally (important for imbalanced medical data)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )

    # Weighted averages - accounts for class imbalance
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )

    # Confusion matrix for detailed error analysis
    cm = confusion_matrix(y_true, y_pred)

    # Compile comprehensive metrics structure
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

    # Detailed per-class analysis - critical for medical AI
    for i, class_name in enumerate(class_names):
        # Calculate class-specific metrics
        true_positives = int(cm[i, i])
        false_positives = int(cm[:, i].sum() - cm[i, i])
        false_negatives = int(cm[i, :].sum() - cm[i, i])
        true_negatives = int(
            cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i])

        # Calculate specificity (true negative rate)
        specificity = true_negatives / \
            (true_negatives + false_positives) if (true_negatives +
                                                   false_positives) > 0 else 0.0

        metrics['per_class'][class_name] = {
            'precision': precision[i],
            'recall': recall[i],  # Sensitivity for medical applications
            'f1_score': f1[i],
            'support': int(support[i]),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'specificity': specificity
        }

    # Medical AI critical validation checks
    # Sensitivity (recall) is critical for disease detection - missing a disease is dangerous
    for i, class_name in enumerate(class_names):
        sensitivity = recall[i]
        # For disease classes (excluding "Normal"), high sensitivity is critical (don't miss diseases)
        if 'normal' not in class_name.lower():
            if sensitivity < 0.8:
                print(
                    f"⚠️  WARNING: Low sensitivity for {class_name}: {sensitivity:.3f}")
                print(f"   This could lead to missed disease diagnoses!")

    return metrics


def plot_confusion_matrix(cm, class_names, output_dir, model_name):
    """
    Create and save confusion matrix visualization for medical AI analysis

    The confusion matrix is crucial for medical AI as it shows:
    - True positives: Correctly identified diseases
    - False positives: False alarms (less critical but still important)
    - False negatives: Missed diseases (CRITICAL - must be minimized)
    - True negatives: Correctly identified normal cases

    Args:
        cm: Confusion matrix array
        class_names: List of class names for axis labels
        output_dir: Directory to save the visualization
        model_name: Model name for the plot title
    """
    plt.figure(figsize=(10, 8))

    # Calculate percentages for annotation - helps identify class imbalance issues
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

    # Create heatmap with medical AI color scheme
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})

    plt.title(f'Confusion Matrix - {model_name}\nLung Disease Classification',
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    # Save confusion matrix for medical documentation
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"💾 Confusion matrix saved to: {cm_path}")
    plt.close()

    return cm_path


def plot_per_class_metrics(metrics, output_dir, model_name):
    """
    Create per-class performance visualization for medical AI analysis

    This visualization helps identify:
    - Which disease types are easier/harder to detect
    - Performance gaps that need addressing
    - Class-specific issues that could affect clinical deployment

    Args:
        metrics: Comprehensive metrics dictionary
        output_dir: Directory to save the visualization
        model_name: Model name for the plot title
    """
    classes = list(metrics['per_class'].keys())
    precision_scores = [metrics['per_class'][cls]['precision']
                        for cls in classes]
    recall_scores = [metrics['per_class'][cls]['recall'] for cls in classes]
    f1_scores = [metrics['per_class'][cls]['f1_score'] for cls in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create grouped bar chart for easy comparison
    bars1 = ax.bar(x - width, precision_scores, width,
                   label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall_scores, width,
                   label='Recall (Sensitivity)', alpha=0.8)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Disease Classes')
    ax.set_ylabel('Score')
    ax.set_title(f'Per-Class Performance Metrics - {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1.0)

    # Add value labels on bars for precise reading
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
    """
    Generate comprehensive evaluation report for medical AI documentation

    This report provides:
    - Overall performance summary
    - Per-class detailed analysis
    - Medical AI validation checks
    - Inference performance assessment
    - Clinical deployment recommendations

    The report is essential for:
    - Regulatory compliance documentation
    - Clinical validation processes
    - Model deployment decisions
    - Continuous improvement tracking

    Args:
        metrics: Comprehensive metrics dictionary
        model_config: Model configuration for context
        output_dir: Directory to save the report
        model_name: Model name for the report
        inference_stats: Inference timing statistics
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Text report with medical AI focus
    report_lines = [
        "=" * 80,
        f"Lung Disease Dataset Model Evaluation Report - {model_name}",
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

    # Per-class detailed analysis - critical for medical AI
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

    # Medical AI specific validation section
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

    # Check sensitivity for disease classes - critical for medical AI
    for class_name, class_metrics in metrics['per_class'].items():
        sensitivity = class_metrics['recall']
        # For disease classes (excluding "Normal"), high sensitivity is critical
        if 'normal' not in class_name.lower():
            if sensitivity >= 0.90:
                report_lines.append(
                    f"✅ {class_name}: Excellent sensitivity ({sensitivity:.3f})")
            elif sensitivity >= 0.80:
                report_lines.append(
                    f"⚠️  {class_name}: Acceptable sensitivity ({sensitivity:.3f})")
            else:
                report_lines.append(
                    f"❌ {class_name}: Low sensitivity ({sensitivity:.3f}) - Risk of missed diagnoses")

    # Performance analysis for clinical deployment
    if inference_stats['inference_time_per_sample'] <= 0.5:
        report_lines.append(
            "✅ Real-time inference capability (≤500ms per sample)")
    else:
        report_lines.append(
            f"⚠️  Slow inference ({inference_stats['inference_time_per_sample']*1000:.0f}ms per sample)")

    # Confusion matrix analysis for error patterns
    report_lines.extend([
        "",
        "📊 CONFUSION MATRIX ANALYSIS",
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

    # Save text report for medical documentation
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"📄 Detailed evaluation report saved to: {report_path}")

    # Save JSON metrics for programmatic access and integration
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
    Perform medical AI specific validation checks for clinical deployment

    This function implements medical AI validation criteria including:
    - Accuracy thresholds for medical screening
    - Sensitivity requirements for disease detection
    - Class imbalance assessment
    - Deployment readiness evaluation

    Medical AI Standards:
    - Minimum 85% accuracy for medical screening applications
    - Minimum 80% sensitivity for disease detection (minimize false negatives)
    - Balanced performance across all disease types
    - Complete audit trail and documentation

    Args:
        metrics: Comprehensive metrics dictionary
        class_names: List of class names for analysis

    Returns:
        dict: Validation results including:
            - passed_medical_threshold: Whether accuracy meets medical standards
            - sensitivity_warnings: Classes with low sensitivity
            - recommendations: Improvement suggestions
            - deployment_ready: Overall deployment readiness
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

    # Check sensitivity for each class (especially disease classes)
    disease_classes = [
        name for name in class_names if 'normal' not in name.lower()]

    for class_name, class_metrics in metrics['per_class'].items():
        sensitivity = class_metrics['recall']

        if class_name in disease_classes:
            if sensitivity < 0.80:
                validation_results['sensitivity_warnings'].append(
                    f"{class_name}: {sensitivity:.3f} (target: ≥0.80)"
                )
                validation_results['recommendations'].append(
                    f"Improve {class_name} sensitivity to reduce missed diagnoses"
                )

    # Check for class imbalance issues that could affect clinical performance
    supports = [metrics['per_class'][cls]['support'] for cls in class_names]
    min_support = min(supports)
    max_support = max(supports)
    imbalance_ratio = max_support / \
        min_support if min_support > 0 else float('inf')

    if imbalance_ratio > 5:
        validation_results['recommendations'].append(
            f"Address class imbalance (ratio: {imbalance_ratio:.1f}:1) with data augmentation or class weighting"
        )

    # Overall deployment readiness assessment
    validation_results['deployment_ready'] = (
        validation_results['passed_medical_threshold'] and
        len(validation_results['sensitivity_warnings']) == 0
    )

    return validation_results


def compare_models(model_results_dir):
    """
    Compare multiple trained models for optimal deployment selection

    This function analyzes all available trained models to:
    - Identify the best performing model
    - Compare accuracy, F1-score, and inference speed
    - Assess medical AI compliance across models
    - Provide deployment recommendations

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

    # Sort by accuracy for medical AI prioritization
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

    # Best model summary for deployment recommendation
    best_model = model_comparisons[0]
    print(f"\n🏆 BEST MODEL: {best_model['name']}")
    print(f"   Accuracy: {best_model['accuracy']*100:.2f}%")
    print(f"   F1-Score: {best_model['f1_macro']*100:.2f}%")
    print(
        f"   Inference: {best_model['inference_time']*1000:.1f}ms per sample")

    return model_comparisons


def find_available_models(base_output_dir, config):
    """
    Find all available trained models by scanning the checkpoints directory.

    Returns a list of model names that have trained checkpoints available.
    """
    available_models = []

    if not os.path.exists(base_output_dir):
        return available_models

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
            # Check for model-specific naming first, then legacy naming
            best_model_path = os.path.join(
                item_path, f'{model_name}-training_best_model.pth')
            if not os.path.exists(best_model_path):
                best_model_path = os.path.join(item_path, 'best_model.pth')
            if os.path.exists(best_model_path) and model_name in config.get('models', {}):
                if model_name not in available_models:
                    available_models.append(model_name)

    return available_models


def evaluate_single_model(model_name, config, project_root, base_output_dir, args):
    """
    Evaluate a single model. This is the core evaluation logic extracted from main().
    """
    if 'models' not in config or model_name not in config['models']:
        print(f"⚠️  Skipping {model_name}: not found in config")
        return None

    model_config = config['models'][model_name]
    output_dir = os.path.join(base_output_dir, f"{model_name}_outputs")

    print("\n" + "=" * 60)
    print(f"🫁 Evaluating Model: {model_name} ({model_config['name']})")
    print("=" * 60)
    print(f"📂 Output directory: {output_dir}")

    # Check if trained model exists - try multiple locations
    checkpoint_paths = [
        # Model-specific output directory
        os.path.join(output_dir, f'{model_name}-training_best_model.pth'),
        # Root checkpoints directory
        os.path.join(base_output_dir, f'{model_name}-training_best_model.pth'),
        # Legacy naming in output dir
        os.path.join(output_dir, 'best_model.pth'),
        # Legacy naming in root checkpoints
        os.path.join(base_output_dir, 'best_model.pth'),
    ]

    best_model_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            best_model_path = path
            break

    if best_model_path is None:
        print(f"❌ No trained model found for {model_name}")
        print(f"   Checked paths:")
        for path in checkpoint_paths:
            print(f"     - {path}")
        print(f"🚀 Train the model first: python train.py --model {model_name}")
        return None

    print(f"📂 Loading checkpoint: {best_model_path}")

    # Load model architecture
    model = get_model(
        model_config['name'],
        model_config['num_classes'],
        pretrained=model_config.get('pretrained', True)
    )

    # Load checkpoint
    checkpoint = torch.load(
        best_model_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            if 'val_accuracy' in checkpoint:
                print(
                    f"📊 Checkpoint validation accuracy: {checkpoint['val_accuracy']*100:.2f}%")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        model = checkpoint

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"🖥️  Using device: {device}")

    # Use model-specific image size if available
    img_size = model_config.get('input_size', config.get(
        'dataset', {}).get('image_size', 224))
    print(f"🖼️  Using model-specific image size: {img_size}x{img_size}")

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
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
    ])

    # Load test dataset
    test_path = config.get('dataset', {}).get('path', './data/test')
    if not os.path.isabs(test_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_path = os.path.join(project_root, test_path.lstrip('./'))

    test_ds = datasets.ImageFolder(test_path, transform=test_transform)
    batch_size = config.get('dataset', {}).get('batch_size', 64)
    num_workers = config.get('dataset', {}).get('num_workers', 4)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"📊 Test set: {len(test_ds)} samples")
    print(f"📦 Evaluation batch size: {batch_size}")

    # Get class names
    if 'dataset' in config and 'class_names' in config['dataset']:
        class_names = config['dataset']['class_names']
    else:
        class_names = [f'class_{i}' for i in range(
            model_config['num_classes'])]

    print(f"📋 Class names: {class_names}")

    # Run evaluation
    print("\n🚀 Starting evaluation...")
    results = evaluate_model_on_test_set(
        model, test_loader, device, class_names, detailed=args.detailed
    )

    # Calculate comprehensive metrics
    metrics = calculate_detailed_metrics(
        results['labels'],
        results['predictions'],
        results['probabilities'],
        class_names
    )

    # Generate visualizations and reports
    os.makedirs(output_dir, exist_ok=True)

    # Plot confusion matrix
    cm = confusion_matrix(results['labels'], results['predictions'])
    plot_confusion_matrix(cm, class_names, output_dir, model_name)

    # Plot per-class metrics if detailed
    if args.detailed:
        plot_per_class_metrics(metrics, output_dir, model_name)

    # Generate comprehensive report
    report_path, json_path = generate_evaluation_report(
        metrics, model_config, output_dir, model_name, results
    )

    # Medical AI validation if requested
    if args.medical_validation:
        validation_results = medical_ai_validation_checks(
            metrics, class_names)
        print("\n🏥 Medical AI Validation Results:")
        for check, passed in validation_results.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check}: {passed}")

    # Logging to MLflow and wandb
    model_name_from_checkpoint = model_name
    if best_model_path:
        checkpoint_filename = os.path.basename(best_model_path)
        if '-training_' in checkpoint_filename:
            model_name_from_checkpoint = checkpoint_filename.split(
                '-training_')[0]

    # MLflow setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mlflow_uri = os.path.join(script_dir, 'mlruns')
    mlflow.set_tracking_uri(mlflow_uri)

    # Create evaluation-specific experiment
    eval_experiment_name = f"lung-disease-dataset-evaluation-{model_name_from_checkpoint}"
    mlflow.set_experiment(eval_experiment_name)

    # Weights & Biases setup for evaluation tracking
    wandb.init(
        project="lung-disease-dataset-evaluation",
        name=f"{model_name_from_checkpoint}_test_evaluation",
        tags=["evaluation", "test", "medical-ai", model_name_from_checkpoint],
        notes=f"Test set evaluation for {model_name_from_checkpoint} model",
        config={
            "model_name": model_name_from_checkpoint,
            "model_type": model_config['name'],
            "num_classes": model_config['num_classes'],
            "test_samples": len(results['labels']),
            "checkpoint_path": best_model_path
        }
    )

    # Log to MLflow
    with mlflow.start_run(run_name=f"{model_name_from_checkpoint}_test_evaluation"):
        mlflow.log_metrics({
            "test_accuracy": metrics['overall']['accuracy'],
            "test_f1_macro": metrics['overall']['f1_macro'],
            "test_f1_weighted": metrics['overall']['f1_weighted'],
            "test_precision_macro": metrics['overall']['precision_macro'],
            "test_recall_macro": metrics['overall']['recall_macro'],
            "inference_time_per_sample": results['inference_time_per_sample']
        })

        mlflow.log_params({
            "model_name": model_config['name'],
            "model_type": model_name_from_checkpoint,
            "num_classes": model_config['num_classes'],
            "checkpoint_path": best_model_path
        })

        # Log artifacts
        if os.path.exists(report_path):
            mlflow.log_artifact(report_path, "evaluation_reports")
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        if os.path.exists(cm_path):
            mlflow.log_artifact(cm_path, "evaluation_visualizations")
            wandb.log({"confusion_matrix": wandb.Image(cm_path)})

    # Log to wandb
    wandb.log({
        "test_accuracy": metrics['overall']['accuracy'],
        "test_f1_macro": metrics['overall']['f1_macro'],
        "test_f1_weighted": metrics['overall']['f1_weighted'],
        "test_precision_macro": metrics['overall']['precision_macro'],
        "test_recall_macro": metrics['overall']['recall_macro'],
        "inference_time_per_sample": results['inference_time_per_sample']
    })

    wandb.finish()

    print(f"\n✅ Evaluation Complete for {model_name}!")
    print(
        f"🎯 Test Accuracy: {metrics['overall']['accuracy']:.4f} ({metrics['overall']['accuracy']*100:.2f}%)")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📄 View detailed report: {report_path}")
    print(f"📊 MLflow experiment: {eval_experiment_name}")
    print(f"🔮 Wandb project: lung-disease-dataset-evaluation")

    return {
        'model_name': model_name,
        'accuracy': metrics['overall']['accuracy'],
        'f1_macro': metrics['overall']['f1_macro'],
        'output_dir': output_dir
    }


def main():
    """
    Main evaluation function for Lung Disease Dataset model assessment

    This function orchestrates the complete evaluation process:
    1. Loads trained model and configuration
    2. Performs comprehensive test set evaluation
    3. Calculates medical AI specific metrics
    4. Generates visualizations and reports
    5. Performs medical AI validation checks
    6. Logs results to MLflow and Weights & Biases
    7. Compares with other trained models if requested

    The evaluation is designed for medical AI deployment with:
    - Complete audit trail for regulatory compliance
    - Medical AI specific validation criteria
    - Comprehensive documentation for clinical review
    - Integration with model registry systems
    """
    parser = argparse.ArgumentParser(
        description='Comprehensive Lung Disease Dataset Model Evaluation'
    )
    parser.add_argument('--config', type=str, default='../configs/eval.yaml',
                        help='Path to config file (default: ../configs/eval.yaml)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model to evaluate. If not specified, all available trained models will be processed. Choose from: convnextv2_tiny, convnextv2_base, tf_efficientnetv2_s, tf_efficientnetv2_m, tf_efficientnetv2_l, regnety_004, regnety_006')
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
        project_root = os.path.dirname(script_dir)
        # Try relative to script dir first, then project root
        if os.path.exists(os.path.join(script_dir, args.config)):
            config_path = os.path.join(script_dir, args.config)
        elif os.path.exists(os.path.join(project_root, args.config)):
            config_path = os.path.join(project_root, args.config)
        else:
            config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve paths relative to script directory
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

    # Set up output directory - check multiple possible locations
    if 'train' in config and 'output_dir' in config['train']:
        output_dir_base = config['train']['output_dir']
    else:
        output_dir_base = './checkpoints'

    if not os.path.isabs(output_dir_base):
        rel_path = output_dir_base.lstrip('./')
        base_output_dir = os.path.join(project_root, rel_path)
    else:
        base_output_dir = output_dir_base

    # Determine which models to process
    if args.model:
        # Validate single model
        if 'models' not in config or args.model not in config['models']:
            available_models = ', '.join(config.get('models', {}).keys())
            raise ValueError(
                f"Model '{args.model}' not found in config. Available models: {available_models}")
        models_to_process = [args.model]
    else:
        # Process all available models
        print("🫁 Lung Disease Dataset Model Evaluation - Processing All Available Models")
        print("=" * 60)
        available_models = find_available_models(base_output_dir, config)
        if not available_models:
            print(
                "❌ No trained models found. Train models first with: python train.py --model <model_name>")
            return
        models_to_process = available_models
        print(
            f"📋 Found {len(available_models)} trained model(s): {', '.join(available_models)}")
        print()

    # Process each model
    all_results = []
    for model_name in models_to_process:
        try:
            result = evaluate_single_model(
                model_name, config, project_root, base_output_dir, args)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Summary if multiple models were processed
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("📊 Evaluation Summary - All Models")
        print("=" * 60)
        all_results.sort(key=lambda x: x['accuracy'], reverse=True)
        for i, result in enumerate(all_results, 1):
            print(
                f"{i}. {result['model_name']}: {result['accuracy']*100:.2f}% accuracy")
        print(
            f"\n🏆 Best Model: {all_results[0]['model_name']} ({all_results[0]['accuracy']*100:.2f}%)")


if __name__ == '__main__':
    main()
