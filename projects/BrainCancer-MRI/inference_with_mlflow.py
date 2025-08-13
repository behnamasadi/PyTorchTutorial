#!/usr/bin/env python3
"""
Inference script using MLflow registered models
This script demonstrates how to load and use models from the MLflow Model Registry
"""

import mlflow
import mlflow.pytorch
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import os


def load_model_from_registry(model_name, stage="Production"):
    """
    Load a model from MLflow Model Registry

    Args:
        model_name (str): Name of the registered model
        stage (str): Model stage (Production, Staging, Archived)

    Returns:
        model: Loaded PyTorch model
    """
    try:
        # Load model from registry
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pytorch.load_model(model_uri)
        print(f"✅ Successfully loaded model: {model_name} ({stage})")
        return model
    except Exception as e:
        print(f"❌ Failed to load model {model_name} ({stage}): {e}")
        return None


def preprocess_image(image_path, img_size=224):
    """
    Preprocess image for model inference

    Args:
        image_path (str): Path to the image file
        img_size (int): Target image size

    Returns:
        tensor: Preprocessed image tensor
    """
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # Convert to 3-channel RGB
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension


def predict(model, image_tensor, class_names=None):
    """
    Make prediction using the loaded model

    Args:
        model: PyTorch model
        image_tensor: Preprocessed image tensor
        class_names (list): List of class names

    Returns:
        dict: Prediction results
    """
    model.eval()
    with torch.no_grad():
        # Forward pass
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

        # Get prediction
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

        # Get all class probabilities
        all_probs = probabilities[0].tolist()

        # Default class names if not provided
        if class_names is None:
            class_names = ["glioma_tumor",
                           "meningioma_tumor", "pituitary_tumor"]

        return {
            "predicted_class": predicted_class,
            "predicted_label": class_names[predicted_class],
            "confidence": confidence,
            "all_probabilities": dict(zip(class_names, all_probs))
        }


def main():
    parser = argparse.ArgumentParser(
        description='Inference using MLflow registered models')
    parser.add_argument('--model', type=str, default='brain-cancer-resnet18',
                        help='Name of the registered model')
    parser.add_argument('--stage', type=str, default='Production',
                        help='Model stage (Production, Staging, Archived)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image file for prediction')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')

    args = parser.parse_args()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    print("🧠 Brain Cancer MRI Inference with MLflow")
    print("=" * 50)

    # Load model from registry
    model = load_model_from_registry(args.model, args.stage)
    if model is None:
        print("❌ Could not load model. Exiting.")
        return

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Image file not found: {args.image}")
        return

    # Preprocess image
    print(f"📸 Loading and preprocessing image: {args.image}")
    try:
        image_tensor = preprocess_image(args.image, args.img_size)
        print(
            f"✅ Image preprocessed successfully (shape: {image_tensor.shape})")
    except Exception as e:
        print(f"❌ Failed to preprocess image: {e}")
        return

    # Make prediction
    print("🔮 Making prediction...")
    try:
        result = predict(model, image_tensor)

        print("\n📊 PREDICTION RESULTS")
        print("-" * 30)
        print(f"🎯 Predicted Class: {result['predicted_label']}")
        print(f"📈 Confidence: {result['confidence']:.2%}")
        print(f"🔢 Class Index: {result['predicted_class']}")

        print("\n📋 All Class Probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob:.2%}")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")


if __name__ == "__main__":
    main()
