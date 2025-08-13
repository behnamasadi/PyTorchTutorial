#!/usr/bin/env python3
"""
Comprehensive MLflow Model Registration Examples
This script demonstrates the concepts mentioned in the MLflow documentation:
- Logging PyTorch models with .pt files internally
- Exporting to ONNX and logging ONNX models
- Registering both model flavors in the Model Registry
- Understanding the difference between logging and registering
"""

import mlflow
import mlflow.pytorch
import mlflow.onnx
import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

class SimpleNet(nn.Module):
    """Simple neural network for demonstration"""
    def __init__(self, input_size=10, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def demonstrate_pytorch_logging_and_registration():
    """
    Demonstrate PyTorch model logging and registration
    This shows how MLflow internally stores .pt files
    """
    print("🔧 PyTorch Model Logging and Registration Example")
    print("=" * 60)
    
    # Create and "train" a simple model
    model = SimpleNet(input_size=10, output_size=3)
    
    # Simulate training (just for demonstration)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Dummy training loop
    for epoch in range(5):
        optimizer.zero_grad()
        dummy_input = torch.randn(32, 10)
        dummy_target = torch.randint(0, 3, (32,))
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
    
    print(f"✅ Model 'trained' (simulated)")
    
    # Start MLflow run
    with mlflow.start_run(run_name="pytorch_model_example"):
        # Log parameters
        mlflow.log_params({
            "model_type": "pytorch",
            "input_size": 10,
            "output_size": 3,
            "epochs": 5
        })
        
        # Log metrics
        mlflow.log_metrics({
            "final_loss": loss.item(),
            "accuracy": 0.85  # dummy accuracy
        })
        
        # Create input example for model signature
        input_example = torch.randn(1, 10)
        
        # LOGGING: Save the model + metadata + artifacts into MLflow
        print("📝 Logging PyTorch model to MLflow...")
        mlflow.pytorch.log_model(
            model, 
            artifact_path="model",
            input_example=input_example,
            registered_model_name="example-pytorch-model"
        )
        
        print("✅ PyTorch model logged successfully!")
        print("📁 Inside MLflow's artifact store, you'll see:")
        print("   model/")
        print("     MLmodel          <-- MLflow model definition")
        print("     conda.yaml       <-- Environment specification")
        print("     pytorch_model.bin <-- Your .pt file (internal)")
        print("     input_example.json <-- Input example for signature")
        
        # REGISTERING: Make this saved model an official, versioned asset
        print("\n🏷️  Model automatically registered in Model Registry!")
        print("📊 Model name: example-pytorch-model")
        print("🔢 Version: 1 (or next available)")
        print("📋 Stage: None (ready for promotion)")

def demonstrate_onnx_export_and_registration():
    """
    Demonstrate ONNX export and registration
    This shows how to export PyTorch to ONNX and register both
    """
    print("\n🔧 ONNX Export and Registration Example")
    print("=" * 60)
    
    # Create the same model
    model = SimpleNet(input_size=10, output_size=3)
    
    # Create input example
    input_example = torch.randn(1, 10)
    
    with mlflow.start_run(run_name="onnx_model_example"):
        # Log PyTorch model first
        print("📝 Logging PyTorch model...")
        mlflow.pytorch.log_model(
            model,
            artifact_path="pytorch_model",
            input_example=input_example,
            registered_model_name="example-pytorch-onnx-comparison"
        )
        
        # Export to ONNX
        print("📦 Exporting to ONNX...")
        onnx_path = "temp_model.onnx"
        torch.onnx.export(
            model,
            input_example,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}}
        )
        
        # Log ONNX model
        print("📝 Logging ONNX model...")
        mlflow.onnx.log_model(
            onnx_path,
            artifact_path="onnx_model",
            input_example=input_example.numpy(),
            registered_model_name="example-onnx-model"
        )
        
        # Clean up temporary file
        os.remove(onnx_path)
        
        print("✅ Both PyTorch and ONNX models logged!")
        print("\n📁 PyTorch artifact store structure:")
        print("   pytorch_model/")
        print("     MLmodel")
        print("     conda.yaml")
        print("     pytorch_model.bin  <-- .pt file")
        print("     input_example.json")
        
        print("\n📁 ONNX artifact store structure:")
        print("   onnx_model/")
        print("     MLmodel")
        print("     conda.yaml")
        print("     model.onnx         <-- .onnx file")
        print("     input_example.json")

def demonstrate_model_flavor_flexibility():
    """
    Demonstrate that MLflow can handle different model flavors
    """
    print("\n🔧 Model Flavor Flexibility Example")
    print("=" * 60)
    
    # Create a simple model
    model = SimpleNet(input_size=5, output_size=2)
    input_example = torch.randn(1, 5)
    
    with mlflow.start_run(run_name="flavor_flexibility_example"):
        # Log PyTorch model
        mlflow.pytorch.log_model(
            model,
            artifact_path="pytorch_flavor",
            input_example=input_example,
            registered_model_name="flavor-example-pytorch"
        )
        
        # Export and log ONNX
        onnx_path = "temp_flavor.onnx"
        torch.onnx.export(model, input_example, onnx_path)
        mlflow.onnx.log_model(
            onnx_path,
            artifact_path="onnx_flavor",
            input_example=input_example.numpy(),
            registered_model_name="flavor-example-onnx"
        )
        os.remove(onnx_path)
        
        print("✅ Multiple model flavors registered:")
        print("   - PyTorch (.pt internally)")
        print("   - ONNX (.onnx)")
        print("\n💡 MLflow doesn't care about the underlying file format!")
        print("   It just needs to be logged in a flavor it understands.")

def demonstrate_registry_metadata():
    """
    Show what the registry tracks
    """
    print("\n🔧 Model Registry Metadata Example")
    print("=" * 60)
    
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    
    # List registered models
    print("📋 Registered Models in Registry:")
    for rm in client.list_registered_models():
        print(f"   - {rm.name}")
        versions = client.get_latest_versions(rm.name, stages=["None", "Staging", "Production"])
        for version in versions:
            print(f"     Version {version.version}: {version.current_stage}")
    
    print("\n📊 Registry tracks:")
    print("   ✅ Model name")
    print("   ✅ Model version")
    print("   ✅ Stage (Staging, Production, Archived)")
    print("   ✅ Metadata, metrics, tags")
    print("   ✅ Path to underlying artifacts")

def main():
    """Run all demonstrations"""
    print("🚀 MLflow Model Registration Concepts Demonstration")
    print("=" * 80)
    print("This script demonstrates the key concepts from the MLflow documentation:")
    print("1. Logging vs Registering")
    print("2. PyTorch model internal .pt storage")
    print("3. ONNX export and registration")
    print("4. Model flavor flexibility")
    print("5. Registry metadata tracking")
    print("=" * 80)
    
    try:
        # Test connection first
        client = mlflow.tracking.MlflowClient()
        client.list_experiments()
        print("✅ Connected to MLflow server")
    except Exception as e:
        print(f"❌ Failed to connect to MLflow server: {e}")
        print("💡 Make sure the server is running with: ./start_mlflow_server.sh")
        return
    
    # Run demonstrations
    demonstrate_pytorch_logging_and_registration()
    demonstrate_onnx_export_and_registration()
    demonstrate_model_flavor_flexibility()
    demonstrate_registry_metadata()
    
    print("\n" + "=" * 80)
    print("🎉 All demonstrations completed!")
    print("🔗 View results at: http://127.0.0.1:5000")
    print("📋 Check the 'Models' tab to see registered models")
    print("📁 Check the 'Experiments' tab to see logged runs")

if __name__ == "__main__":
    main()
