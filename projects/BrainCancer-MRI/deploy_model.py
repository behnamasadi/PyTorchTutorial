#!/usr/bin/env python3
"""
Brain Cancer MRI Model Deployment Script

Usage:
    python3 deploy_model.py --model efficientnet_b0 --version 1.0.0
"""

import argparse
import torch
import mlflow.pytorch
import os


def main():
    parser = argparse.ArgumentParser(
        description='Deploy Brain Cancer MRI model')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--version', type=str,
                        default='1', help='Model version')

    args = parser.parse_args()

    print("🏥 Brain Cancer MRI Model Deployment")
    print("=" * 50)
    print(f"📋 Model: {args.model}")
    print(f"🏷️  Version: {args.version}")

    try:
        # Load from MLflow Model Registry
        model_uri = f"models:/brain-cancer-mri-{args.model}/{args.version}"
        model = mlflow.pytorch.load_model(model_uri)
        print(f"✅ Model loaded successfully from registry!")
        print(f"🔗 Registry URI: {model_uri}")

        # Model info
        print(f"\n📊 **Model Information:**")
        print(f"📦 Type: {type(model).__name__}")
        print(f"🏷️  Version: {args.version}")
        print(f"📋 Registry: brain-cancer-mri-{args.model}")

        print(f"\n🚀 **Ready for Production Deployment!**")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print(f"💡 Make sure the model is registered first:")
        print(
            f"   python3 register_model.py --model {args.model} --version {args.version}")


if __name__ == '__main__':
    main()
