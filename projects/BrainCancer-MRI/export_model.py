import argparse
import yaml
import os
import torch
import torch.onnx
import torchvision.models as models
from models.model import get_model


def export_to_torchscript(model, example_input, output_path, model_name):
    """Export model to TorchScript format"""
    print(f"📦 Exporting {model_name} to TorchScript...")

    model.eval()

    # Trace the model
    try:
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(output_path)
        print(f"✅ TorchScript model saved to: {output_path}")

        # Test the exported model
        test_output = traced_model(example_input)
        print(
            f"🧪 TorchScript test successful - output shape: {test_output.shape}")

        return True
    except Exception as e:
        print(f"❌ TorchScript export failed: {e}")
        return False


def export_to_onnx(model, example_input, output_path, model_name, input_names=None, output_names=None):
    """Export model to ONNX format"""
    print(f"📦 Exporting {model_name} to ONNX...")

    model.eval()

    input_names = input_names or ['input']
    output_names = output_names or ['output']

    try:
        torch.onnx.export(
            model,
            example_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✅ ONNX model saved to: {output_path}")
        return True
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return False


def get_model_info_for_export(model_name):
    """Get export-specific information for each model"""
    info = {
        'resnet18': {'input_size': (1, 3, 224, 224), 'description': 'ResNet-18 CNN'},
        'resnet50': {'input_size': (1, 3, 224, 224), 'description': 'ResNet-50 CNN'},
        'efficientnet_b0': {'input_size': (1, 3, 224, 224), 'description': 'EfficientNet-B0'},
        'swin_t': {'input_size': (1, 3, 224, 224), 'description': 'Swin Transformer Tiny'},
        'swin_s': {'input_size': (1, 3, 224, 224), 'description': 'Swin Transformer Small'},
        'vit_b_16': {'input_size': (1, 3, 224, 224), 'description': 'Vision Transformer'},
        'medical_cnn': {'input_size': (1, 3, 224, 224), 'description': 'Medical CNN'},
        'xception_medical': {'input_size': (1, 3, 299, 299), 'description': 'Xception Medical (Kaggle)'}
    }
    return info.get(model_name, {'input_size': (1, 3, 224, 224), 'description': 'Unknown Model'})


def main(config_path, model_name, export_format='both'):
    # Resolve absolute path to config file
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get selected model (override or from config)
    selected_model = model_name if model_name else config['model']

    if selected_model not in config['models']:
        available_models = ', '.join(config['models'].keys())
        raise ValueError(
            f"Model '{selected_model}' not found. Available: {available_models}")

    model_config = config['models'][selected_model]

    # Create model-specific output directory
    if not os.path.isabs(config['train']['output_dir']):
        rel_path = config['train']['output_dir'].lstrip('./')
        base_output_dir = os.path.join(script_dir, rel_path)
    else:
        base_output_dir = config['train']['output_dir']

    model_output_dir = os.path.join(
        base_output_dir, f"{selected_model}_outputs")

    print(f"🚀 **Model Export for Deployment**")
    print("=" * 50)
    print(f"📋 Model: {model_config['name']}")
    print(f"📂 Checkpoint directory: {model_output_dir}")

    # Load trained model
    checkpoint_path = os.path.join(model_output_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"❌ No trained model found at {checkpoint_path}. Train the model first with: python3 train.py --model {selected_model}")

    print(f"📂 Loading checkpoint: {checkpoint_path}")

    # Load model architecture
    weights = eval(model_config['weights']
                   ) if model_config['weights'] else None
    model, _ = get_model(
        model_config['name'],
        model_config['num_classes'],
        weights
    )

    # Load trained weights
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(
            f"🎯 Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(
            f"📊 Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.4f}")
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    # Set to evaluation mode
    model.eval()

    # Get model-specific input size
    model_info = get_model_info_for_export(selected_model)
    input_size = model_info['input_size']

    print(f"🖼️  Input size: {input_size[2]}x{input_size[3]}")

    # Create example input
    example_input = torch.randn(*input_size)

    # Create export directory
    export_dir = os.path.join(model_output_dir, 'exported_models')
    os.makedirs(export_dir, exist_ok=True)

    # Export to different formats
    success_count = 0

    if export_format in ['torchscript', 'both']:
        torchscript_path = os.path.join(
            export_dir, f"{selected_model}_model.pt")
        if export_to_torchscript(model, example_input, torchscript_path, selected_model):
            success_count += 1

    if export_format in ['onnx', 'both']:
        onnx_path = os.path.join(export_dir, f"{selected_model}_model.onnx")
        if export_to_onnx(model, example_input, onnx_path, selected_model):
            success_count += 1

    print(f"\n🎉 **Export Complete!**")
    print(f"📁 Exported models saved to: {export_dir}")
    print(f"✅ {success_count} format(s) exported successfully")

    # Show file sizes
    print(f"\n📊 **Exported Model Files:**")
    for file in os.listdir(export_dir):
        file_path = os.path.join(export_dir, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  📄 {file}: {size_mb:.2f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export trained Brain Cancer MRI model for deployment')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to export. Choose from: resnet18, swin_t, swin_s, efficientnet_b0, vit_b_16, medical_cnn, xception_medical')
    parser.add_argument('--format', type=str, choices=['torchscript', 'onnx', 'both'], default='both',
                        help='Export format (default: both)')
    args = parser.parse_args()
    main(args.config, args.model, args.format)
