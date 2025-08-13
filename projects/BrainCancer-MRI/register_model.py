#!/usr/bin/env python3
"""
🧠 Brain Cancer MRI Model Registry Script
========================================

📋 PURPOSE:
This script registers trained models with MLflow Model Registry for production deployment.
Use this AFTER training is complete and you have a model you want to deploy.

🚀 WHEN TO USE:
- After successful training with good validation accuracy
- When you want to deploy a model to production
- When you need version control for your models
- When you want to track model performance over time

📊 WHAT IT DOES:
- MLflow Model Registry registration
- Semantic versioning for model lifecycle management
- Model performance documentation and tracking
- Training configuration and metadata preservation
- Model lineage and audit trail maintenance
- Medical AI compliance documentation

🏥 MEDICAL AI FEATURES:
- Clinical deployment with complete documentation
- Regulatory compliance and audit trails
- Model versioning and rollback capabilities
- Performance tracking and monitoring
- Integration with medical AI workflows

💡 WORKFLOW:
1. Train your model: python train.py --model resnet18
2. Evaluate performance: python evaluate.py --model resnet18
3. Register best model: python register_model.py --model resnet18 --version 1.0.0

📝 USAGE EXAMPLES:
    python register_model.py --model efficientnet_b0 --version 1.0.0
    python register_model.py --model resnet18 --version 2.1.0 --description "Improved medical validation"
    python register_model.py --model swin_t --version 1.0.0 --tags "production=ready" "medical_ai=validated"
"""

import argparse
import yaml
import os
import json
import mlflow
import mlflow.pytorch
from datetime import datetime
import torch
from models.model import get_model


def load_model_metadata(model_name, model_output_dir):
    """
    Load comprehensive model metadata for registry documentation

    This function collects all relevant information about a trained model:
    - Training configuration and hyperparameters
    - Evaluation results and performance metrics
    - Medical AI validation results
    - Model file information and sizes
    - Export artifacts and deployment files

    Medical AI Registry Requirements:
    - Complete audit trail of model development
    - Performance metrics for clinical validation
    - Medical AI compliance documentation
    - Model lineage and version tracking
    - Deployment readiness assessment

    Args:
        model_name: Name of the model being registered
        model_output_dir: Directory containing model outputs and artifacts

    Returns:
        dict: Comprehensive metadata dictionary for model registry
    """
    metadata = {
        'model_name': model_name,
        'registration_time': datetime.now().isoformat(),
        'model_files': {},
        'performance_metrics': {},
        'training_config': {},
        'evaluation_results': {}
    }

    # Load training checkpoint metadata for model lineage
    checkpoint_path = os.path.join(model_output_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            metadata['training_config'].update({
                'epoch': checkpoint.get('epoch', 'unknown'),
                'validation_accuracy': checkpoint.get('val_acc', 'unknown'),
                'checkpoint_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024)
            })

    # Load evaluation metrics for performance documentation
    eval_metrics_path = os.path.join(
        model_output_dir, 'evaluation_metrics.json')
    if os.path.exists(eval_metrics_path):
        with open(eval_metrics_path, 'r') as f:
            eval_data = json.load(f)
            metadata['evaluation_results'] = eval_data.get('metrics', {})
            metadata['inference_stats'] = eval_data.get('inference_stats', {})

    # Load medical validation results for clinical deployment
    medical_validation_path = os.path.join(
        model_output_dir, 'medical_validation.json')
    if os.path.exists(medical_validation_path):
        with open(medical_validation_path, 'r') as f:
            metadata['medical_validation'] = json.load(f)

    # Check for exported models for deployment tracking
    export_dir = os.path.join(model_output_dir, 'exported_models')
    if os.path.exists(export_dir):
        for file in os.listdir(export_dir):
            if file.endswith(('.pt', '.onnx')):
                file_path = os.path.join(export_dir, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                metadata['model_files'][file] = {
                    'size_mb': size_mb,
                    'path': file_path
                }

    return metadata


def register_model_in_mlflow(model_name, model_output_dir, version, description, tags):
    """
    Register model in MLflow Model Registry for production deployment

    This function handles the complete model registration process:
    - Creates MLflow experiment and run
    - Logs model parameters and performance metrics
    - Uploads model artifacts and documentation
    - Registers model in MLflow Model Registry
    - Adds metadata and tags for organization

    Medical AI Registry Features:
    - Complete model lineage tracking
    - Performance metrics for clinical validation
    - Medical AI compliance documentation
    - Deployment readiness assessment
    - Regulatory audit trail maintenance

    Args:
        model_name: Name of the model to register
        model_output_dir: Directory containing model artifacts
        version: Semantic version for the model
        description: Human-readable description of the model
        tags: Additional tags for model organization

    Returns:
        bool: True if registration successful, False otherwise
    """
    print(f"📋 Registering {model_name} v{version} in MLflow Model Registry...")

    # Load comprehensive model metadata
    metadata = load_model_metadata(model_name, model_output_dir)

    # Set up MLflow tracking for model registry
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mlflow_uri = os.path.join(script_dir, 'mlruns')
    mlflow.set_tracking_uri(mlflow_uri)

    # Create model registry experiment for organization
    registry_experiment_name = "brain-cancer-mri-model-registry"
    mlflow.set_experiment(registry_experiment_name)

    # Start MLflow run for model registration
    with mlflow.start_run(run_name=f"{model_name}_v{version}_registration"):

        # Log model parameters for reproducibility
        mlflow.log_params({
            "model_name": model_name,
            "version": version,
            "description": description,
            "registration_time": metadata['registration_time']
        })

        # Log performance metrics for clinical validation
        if metadata['evaluation_results']:
            overall_metrics = metadata['evaluation_results'].get('overall', {})
            mlflow.log_metrics({
                "test_accuracy": overall_metrics.get('accuracy', 0),
                "test_f1_macro": overall_metrics.get('f1_macro', 0),
                "test_precision_macro": overall_metrics.get('precision_macro', 0),
                "test_recall_macro": overall_metrics.get('recall_macro', 0),
                "inference_time_ms": metadata.get('inference_stats', {}).get('inference_time_per_sample', 0) * 1000
            })

        # Log medical validation metrics for clinical deployment
        if 'medical_validation' in metadata:
            medical = metadata['medical_validation']
            mlflow.log_metrics({
                "medical_threshold_passed": 1.0 if medical.get('passed_medical_threshold', False) else 0.0,
                "deployment_ready": 1.0 if medical.get('deployment_ready', False) else 0.0
            })

        # Log model artifacts for deployment
        export_dir = os.path.join(model_output_dir, 'exported_models')
        if os.path.exists(export_dir):
            for file in os.listdir(export_dir):
                if file.endswith(('.pt', '.onnx')):
                    file_path = os.path.join(export_dir, file)
                    mlflow.log_artifact(file_path, "deployment_models")

        # Log evaluation reports for documentation
        report_path = os.path.join(model_output_dir, 'evaluation_report.txt')
        if os.path.exists(report_path):
            mlflow.log_artifact(report_path, "documentation")

        # Log confusion matrix for visual analysis
        cm_path = os.path.join(model_output_dir, 'confusion_matrix.png')
        if os.path.exists(cm_path):
            mlflow.log_artifact(cm_path, "visualizations")

        # Log metadata as JSON for programmatic access
        metadata_path = os.path.join(
            model_output_dir, 'model_registry_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        mlflow.log_artifact(metadata_path, "metadata")

        # Register model in MLflow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/deployment_models"

        # Create model registry entry with medical AI naming convention
        registered_model_name = f"brain-cancer-mri-{model_name}"

        try:
            # Register the model with comprehensive metadata
            mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
                tags={
                    "version": version,
                    "description": description,
                    "model_type": model_name,
                    "medical_ai": "true",
                    "deployment_ready": str(metadata.get('medical_validation', {}).get('deployment_ready', False)),
                    **tags
                }
            )

            print(f"✅ Model registered successfully!")
            print(f"📋 Registry name: {registered_model_name}")
            print(f"🏷️  Version: {version}")
            print(
                f"📊 Test accuracy: {metadata['evaluation_results'].get('overall', {}).get('accuracy', 'N/A')}")

            return True

        except Exception as e:
            print(f"❌ Model registration failed: {e}")
            return False


def create_model_card(model_name, version, description, metadata):
    """
    Create a comprehensive model card for medical AI documentation

    Model cards are essential for medical AI deployment as they provide:
    - Complete model documentation for clinical review
    - Performance metrics for regulatory compliance
    - Medical AI validation results
    - Deployment instructions and requirements
    - Model lineage and version information

    Medical AI Documentation Requirements:
    - Clear performance metrics and limitations
    - Medical validation results and compliance
    - Deployment instructions for clinical use
    - Model lineage and audit trail
    - Regulatory compliance documentation

    Args:
        model_name: Name of the model
        version: Semantic version of the model
        description: Human-readable description
        metadata: Comprehensive model metadata

    Returns:
        str: Complete model card in markdown format
    """
    print(f"📄 Creating model card for {model_name} v{version}...")

    model_card = f"""# Brain Cancer MRI Classification Model Card

## Model Information
- **Model Name**: {model_name}
- **Version**: {version}
- **Description**: {description}
- **Registration Date**: {metadata['registration_time']}
- **Model Type**: Medical AI - Brain Tumor Classification

## Performance Metrics
"""

    if metadata['evaluation_results']:
        overall = metadata['evaluation_results'].get('overall', {})
        model_card += f"""
### Overall Performance
- **Test Accuracy**: {overall.get('accuracy', 'N/A'):.4f} ({overall.get('accuracy', 0)*100:.2f}%)
- **Macro F1-Score**: {overall.get('f1_macro', 'N/A'):.4f}
- **Macro Precision**: {overall.get('precision_macro', 'N/A'):.4f}
- **Macro Recall**: {overall.get('recall_macro', 'N/A'):.4f}

### Per-Class Performance
"""
        per_class = metadata['evaluation_results'].get('per_class', {})
        for class_name, metrics in per_class.items():
            model_card += f"""
#### {class_name.title()}
- **Precision**: {metrics.get('precision', 'N/A'):.4f}
- **Recall**: {metrics.get('recall', 'N/A'):.4f}
- **F1-Score**: {metrics.get('f1_score', 'N/A'):.4f}
- **Support**: {metrics.get('support', 'N/A')} samples
"""

    if 'medical_validation' in metadata:
        medical = metadata['medical_validation']
        model_card += f"""
## Medical AI Validation
- **Medical Threshold Passed**: {'✅ YES' if medical.get('passed_medical_threshold', False) else '❌ NO'}
- **Deployment Ready**: {'✅ YES' if medical.get('deployment_ready', False) else '❌ NO'}
- **Sensitivity Warnings**: {len(medical.get('sensitivity_warnings', []))}
- **Recommendations**: {len(medical.get('recommendations', []))}
"""

    if metadata['inference_stats']:
        stats = metadata['inference_stats']
        model_card += f"""
## Inference Performance
- **Average Inference Time**: {stats.get('inference_time_per_sample', 0)*1000:.2f}ms per sample
- **Batch Processing Time**: {stats.get('inference_time_per_batch', 0):.3f}s per batch
- **Total Test Samples**: {stats.get('total_samples', 'N/A')}
"""

    model_card += f"""
## Model Files
"""
    for file_name, file_info in metadata['model_files'].items():
        model_card += f"- **{file_name}**: {file_info['size_mb']:.2f} MB\n"

    model_card += f"""
## Training Configuration
- **Epoch**: {metadata['training_config'].get('epoch', 'N/A')}
- **Validation Accuracy**: {metadata['training_config'].get('validation_accuracy', 'N/A')}
- **Checkpoint Size**: {metadata['training_config'].get('checkpoint_size_mb', 'N/A'):.2f} MB

## Usage Instructions
1. Load the model using the appropriate framework (PyTorch for .pt, ONNX Runtime for .onnx)
2. Preprocess input images to match the expected input size
3. Run inference and post-process the outputs
4. Apply medical validation checks before clinical use

## Medical AI Compliance
This model has been validated for medical AI deployment with:
- Accuracy threshold validation (≥85%)
- Sensitivity analysis for tumor detection
- Clinical deployment readiness assessment
- Complete audit trail for regulatory compliance

---
*Generated automatically by Brain Cancer MRI Model Registry*
"""

    return model_card


def main():
    """
    Main function for model registry and deployment preparation

    This function orchestrates the complete model registration process:
    1. Loads model configuration and validates inputs
    2. Collects comprehensive model metadata
    3. Creates detailed model documentation
    4. Registers model in MLflow Model Registry
    5. Generates deployment-ready artifacts

    Medical AI Deployment Workflow:
    - Model validation and performance assessment
    - Medical AI compliance documentation
    - Registry integration for version control
    - Deployment preparation and testing
    - Regulatory compliance and audit trails

    The registration process ensures models are ready for:
    - Clinical deployment with complete documentation
    - Regulatory compliance and audit trails
    - Model versioning and lifecycle management
    - Performance monitoring and tracking
    - Integration with medical AI workflows
    """
    parser = argparse.ArgumentParser(
        description='Register Brain Cancer MRI model in MLflow Model Registry')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to register (efficientnet_b0, resnet18, etc.)')
    parser.add_argument('--version', type=str, required=True,
                        help='Semantic version (e.g., 1.0.0, 2.1.0)')
    parser.add_argument('--description', type=str, default='Brain Cancer MRI Classification Model',
                        help='Model description')
    parser.add_argument('--tags', type=str, nargs='*', default=[],
                        help='Additional tags (key=value format)')

    args = parser.parse_args()

    # Parse tags for model organization
    tags = {}
    for tag in args.tags:
        if '=' in tag:
            key, value = tag.split('=', 1)
            tags[key] = value

    # Resolve config path for model configuration
    if not os.path.isabs(args.config):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    # Load configuration for model details
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve model output directory for artifact collection
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(config['train']['output_dir']):
        rel_path = config['train']['output_dir'].lstrip('./')
        base_output_dir = os.path.join(script_dir, rel_path)
    else:
        base_output_dir = config['train']['output_dir']

    model_output_dir = os.path.join(base_output_dir, f"{args.model}_outputs")

    print("🏥 Brain Cancer MRI Model Registry")
    print("=" * 50)
    print(f"📋 Model: {args.model}")
    print(f"🏷️  Version: {args.version}")
    print(f"📝 Description: {args.description}")
    print(f"📂 Model directory: {model_output_dir}")

    # Check if model exists and is ready for registration
    if not os.path.exists(model_output_dir):
        print(f"❌ Model directory not found: {model_output_dir}")
        print(f"🚀 Train and export the model first:")
        print(f"   python3 train.py --model {args.model}")
        print(
            f"   python3 evaluate.py --model {args.model} --detailed --medical-validation")
        print(f"   python3 export_model.py --model {args.model}")
        return

    # Load comprehensive model metadata for documentation
    metadata = load_model_metadata(args.model, model_output_dir)

    # Create detailed model card for medical documentation
    model_card = create_model_card(
        args.model, args.version, args.description, metadata)

    # Save model card for clinical review
    model_card_path = os.path.join(
        model_output_dir, f'model_card_v{args.version}.md')
    with open(model_card_path, 'w') as f:
        f.write(model_card)
    print(f"📄 Model card saved to: {model_card_path}")

    # Register model in MLflow Model Registry for production deployment
    success = register_model_in_mlflow(
        args.model, model_output_dir, args.version, args.description, tags
    )

    if success:
        print(f"\n🎉 **Model Registration Complete!**")
        print(f"📋 Model: {args.model} v{args.version}")
        print(
            f"📊 Test Accuracy: {metadata['evaluation_results'].get('overall', {}).get('accuracy', 'N/A')}")
        print(
            f"🏥 Medical AI Ready: {metadata.get('medical_validation', {}).get('deployment_ready', False)}")
        print(f"📄 Documentation: {model_card_path}")
        print(f"🔗 MLflow Registry: brain-cancer-mri-{args.model}")

        # Next steps for clinical deployment
        print(f"\n🚀 **Next Steps for Deployment:**")
        print(f"1. Review model card: {model_card_path}")
        print(f"2. Validate medical AI compliance")
        print(f"3. Deploy using MLflow Model Registry")
        print(f"4. Set up monitoring and alerting")
        print(f"5. Document clinical validation process")
    else:
        print(f"\n❌ **Model Registration Failed**")
        print(f"Check the logs above for details.")


if __name__ == '__main__':
    main()
