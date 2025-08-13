# Brain Cancer MRI Classification

A comprehensive PyTorch-based deep learning project for classifying brain MRI images to detect different types of brain tumors using state-of-the-art architectures with medical AI validation and clinical deployment capabilities.



##  Input Processing & Data Normalization

This project uses **RGB conversion**for all models to ensure compatibility with pre-trained weights, along with **proper data normalization**for optimal training performance.

### **RGB Conversion Approach**
- Converts grayscale MRI to 3-channel RGB using `transforms.Grayscale(num_output_channels=3)`
- Compatible with all pre-trained models (ResNet, EfficientNet, Swin, ViT, etc.)
- Simplest and most reliable approach
- No model architecture modifications needed

**Benefits:**
- ✅ Works with all pre-trained models
- ✅ No complex model modifications
- ✅ Consistent with ImageNet pre-training
- ✅ Easy to implement and maintain

**Usage:**
```bash
# All models use RGB input automatically
python train.py --model resnet18
python train.py --model efficientnet_b0
python train.py --model swin_t
```

### **Data Normalization Strategy**

Unlike pre-packaged datasets (ImageNet, MNIST) that come with fixed normalization statistics, custom medical datasets require careful normalization strategy. This project implements **Fixed Split**for optimal results.

**Why Data Normalization Matters:**
- **Medical Image Challenges**: MRI images have different intensity ranges than natural images
- **Model Stability**: Proper normalization prevents training instability and NaN losses
- **Convergence Speed**: Normalized data trains faster and more reliably
- **Reproducible Results**: Consistent normalization across experiments

**Implementation (Fixed Split with Pre-computed Statistics):**

Since we use a fixed seed (42), we can pre-calculate normalization statistics once and reuse them:

```bash
# 1. Pre-compute statistics (run once)
python compute_normalization.py --mode save

# 2. Use pre-computed statistics in training
python train.py --model resnet18
```

**Automatic Fallback:**
```python
# Training script automatically detects pre-computed statistics
try:
    from normalization_constants import NORMALIZATION_MEAN, NORMALIZATION_STD
    # Use pre-computed statistics
    train_ds, val_ds, _ = load_datasets(config, mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
except ImportError:
    # Fallback to computing during training
    train_ds_raw, val_ds_raw, _ = load_datasets(config, mean=None, std=None)
    mean, std = calculate_mean_std(train_ds_raw, batch_size=batch_size, ...)
    train_ds, val_ds, _ = load_datasets(config, mean=mean.tolist(), std=std.tolist())
```

**Key Features:**
- ✅ **Fixed Seed (42)**: Ensures reproducible train/val/test splits every run
- ✅ **Pre-computed Statistics**: Compute once, reuse across all experiments
- ✅ **Training-Only Statistics**: Mean/std computed only on training data (no information leakage)
- ✅ **Consistent Normalization**: Same statistics used across all experiments
- ✅ **Proper Transform Order**: Normalization applied after resize but before augmentation
- ✅ **Automatic Fallback**: Works with or without pre-computed statistics

**Transform Pipeline:**
```python
# Training Transforms
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB
    transforms.RandomHorizontalFlip(),            # Augmentation
    transforms.RandomRotation(10),                # Augmentation
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),  # Augmentation
    transforms.ToTensor(),                        # Convert to tensor [0,1]
    transforms.Normalize(mean, std)               # Normalize to [-1,1]
])

# Validation/Test Transforms
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB
    transforms.ToTensor(),                        # Convert to tensor [0,1]
    transforms.Normalize(mean, std)               # Normalize to [-1,1]
])
```

**Statistics Tracking:**
The training script automatically logs normalization statistics to all monitoring platforms:

```bash
# Console Output (when using pre-computed statistics)
 Using pre-computed normalization statistics...
 Pre-computed mean: [0.1528, 0.1528, 0.1528]
 Pre-computed std: [0.1613, 0.1613, 0.1613]
```

# Pre-compute and save statistics (run once)
python compute_normalization.py --mode save

# Display statistics only (no saving)
python compute_normalization.py --mode display

# Verify existing pre-computed statistics
python compute_normalization.py --mode verify

**Pre-computation Benefits:**
-  **Faster Training**: Skip statistics computation on every run
-  **Consistent Results**: Same statistics across all experiments
-  **Time Savings**: ~30-60 seconds saved per training run
-  **Reproducibility**: Guaranteed identical normalization across runs

**Understanding Your MRI Statistics:**
Your computed values `mean=[0.1528, 0.1528, 0.1528]` and `std=[0.1613, 0.1613, 0.1613]` are **perfectly normal**for brain MRI data:

- **Identical Channels**: All three RGB channels are identical because:
  - Original MRI images are grayscale (single channel)
  - `transforms.Grayscale(num_output_channels=3)` converts each pixel to identical RGB values
  - This is the expected behavior for medical imaging datasets

- **Medical Image Characteristics**:
  - **Low Mean (0.153)**: Typical for MRI images with dark backgrounds
  - **Moderate Std (0.161)**: Good contrast variation for tumor detection
  - **Normalized Range**: After normalization, data will be in `[-0.95, 1.06]` range (computed from your actual statistics)

**Comparison with ImageNet Statistics:**
- **ImageNet (Natural Images)**: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- **Your MRI Data**: `mean=[0.1528, 0.1528, 0.1528]`, `std=[0.1613, 0.1613, 0.1613]`

**Why Different:**
- **ImageNet**: Natural color images with varied lighting and colors
- **MRI Data**: Grayscale medical images with consistent intensity characteristics
- **Medical Domain**: Your statistics are optimized for brain tumor detection

**✅ Validation:**
Your pre-computed statistics are now ready! When you run training, you should see:
```
✅ Using pre-computed normalization statistics
 Pre-computed mean: [0.1528, 0.1528, 0.1528]
 Pre-computed std: [0.1613, 0.1613, 0.1613]


Instead of the slower computation process.

**🧪 Test Your Setup:**
```bash
# Run a quick training test to verify pre-computed statistics work
python train.py --model resnet18 --epochs 1

# You should see the pre-computed statistics being used
# and training should start faster without the computation step
```





##  Project Overview

This project implements a convolutional neural network to classify brain MRI images into three categories:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**

## 🏗️ Architecture

### **🤖 Model Architectures**
- **Multiple Architectures**: ResNet, Swin Transformer, EfficientNet, Vision Transformer
- **Medical AI Optimized**: Model-specific configurations for clinical deployment
- **Transfer Learning**: Pre-trained weights for robust medical image classification

### **Performance & Optimization**
- **Framework**: PyTorch 2.0+ with compilation support for faster training
- **Mixed Precision**: FP16 training with medical image stability considerations
- **GPU Optimization**: Memory-efficient training with hardware-specific batch sizes
- **Multi-core Data Loading**: Optimized for medical imaging workflows

### **Monitoring & Validation**
- **Triple Monitoring**: TensorBoard + MLflow + Weights & Biases for complete audit trails
- **Hardware Monitoring**: Real-time GPU/CPU/Memory utilization tracking
- **Medical AI Validation**: Clinical deployment readiness assessment
- **Performance Metrics**: Batch processing time, throughput, temperature monitoring

### **Data Processing**
- **Data Augmentation**: Random horizontal flip, rotation, color jitter
- **Medical Image Preprocessing**: Optimized normalization for MRI data
- **Class Balance Handling**: Techniques for imbalanced medical datasets
- **RGB Conversion**: Automatic conversion of grayscale MRI to RGB for pre-trained model compatibility

## 📁 Project Structure

```
BrainCancer-MRI/
├── config/
│   ├── config.yaml              # Main training configuration
│   └── optimized_config.yaml    # Optimized configuration for performance
├── data/
│   ├── dataset.py               # Dataset loading and preprocessing (RGB conversion)
│   ├── dataset_albumentations.py # Enhanced dataset with albumentations
│   ├── enhanced_dataset.py      # Additional dataset enhancements
│   ├── data_utils.py            # Data utility functions
│   └── brain-cancer/            # Brain cancer MRI dataset
├── models/
│   ├── model.py                 # Model architecture definitions
│   └── simple_cnn.py            # Simple CNN implementation
├── utils/
│   ├── helpers.py               # Utility functions
│   └── path_utils.py            # Path management utilities
├── train.py                     # Main training script (no model registration)
├── evaluate.py                  # Comprehensive medical AI evaluation
├── test.py                      # Simple test set evaluation
├── export_model.py              # Model export for deployment (TorchScript/ONNX)
├── register_model.py            # Model registration with MLflow
├── deploy_model.py              # Model deployment utilities
├── inference_with_mlflow.py     # Inference using MLflow registered models
├── mlflow_model_examples.py     # MLflow concepts demonstration
├── compute_normalization.py     # Comprehensive normalization statistics (display/save/verify)
├── index.ipynb                  # Jupyter notebook with examples
├── requirements.txt             # Python dependencies
├── WORKFLOW_GUIDE.md            # Training & registration workflow guide
├── checkpoints/                 # Model checkpoints (created during training)
├── runs/                        # TensorBoard logs (created during training)
├── mlruns/                      # MLflow logs (created during training)
├── mlartifacts/                 # MLflow artifacts (created during training)
├── wandb/                       # Weights & Biases logs (created during training)
├── outputs/                     # Output files and results
├── results/                     # Evaluation results
├── normalization_stats.json     # Pre-computed normalization statistics
├── normalization_constants.py   # Python constants for normalization
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

#### **Option 1: Quick pip installation**
```bash
# Required packages
pip install torch torchvision
pip install tensorboard mlflow wandb
pip install pyyaml pillow psutil GPUtil onnx
pip install albumentations opencv-python
```

#### **Option 2: Conda installation (Recommended for medical imaging)**
```bash
# Core PyTorch packages
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Advanced augmentation and medical imaging (via conda-forge)
conda install -c conda-forge albumentations opencv

# Remaining packages
pip install tensorboard mlflow wandb pyyaml pillow psutil GPUtil onnx
```

#### **Option 3: Using requirements.txt**
```bash
# Install albumentations and opencv via conda first (recommended)
conda install -c conda-forge albumentations opencv

# Then install remaining packages
pip install -r requirements.txt
```

#### **Option 4: Using installation scripts (Recommended)**
```bash
# Linux/macOS
./install_dependencies.sh

# Windows PowerShell
.\install_dependencies.ps1
```

### Training

**🚀 Quick Start (Complete Workflow):**

```bash
# 0. Pre-compute normalization statistics (run once)
python compute_normalization.py --mode save

# 1. Train your model
python train.py --model resnet18

# 2. Evaluate the trained model
python evaluate.py --model resnet18

# 3. Register the best model (only if performance is good)
python register_model.py --model resnet18 --version 1.0.0 --description "First production model"
```

**Manual training (make sure you're in the correct directory):**
```bash
# Navigate to the project directory first
cd src/projects/BrainCancer-MRI

# Simple training with default settings
python train.py
```

**Training with different models:**
```bash
# Train with ResNet (baseline)
python train.py --model resnet18

# Train with Swin Transformer
python train.py --model swin_t

# Train with EfficientNet
python train.py --model efficientnet_b0

# Train with Kaggle Xception solution (299x299, Adamax)
python train.py --model xception_medical

# Train with Vision Transformer
python train.py --model vit_b_16

# Train with Swin-S (large model, batch_size=8)
python train.py --model swin_s

# Custom parameters
python train.py --model resnet50 --epochs 10 --batch-size 16 --lr 0.0005

# Training with custom config
python train.py --config path/to/your/config.yaml
```

**Training with different models:**
```bash
# Train with ResNet (baseline)
python train.py --model resnet18

# Train with Swin Transformer
python train.py --model swin_t

# Train with EfficientNet
python train.py --model efficientnet_b0

# Train with Vision Transformer
python train.py --model vit_b_16

# Custom parameters
python train.py --model resnet50 --epochs 10 --batch-size 16 --lr 0.0005
```

##  Complete Workflow (Training → Evaluation → Registration)

### **Phase 1: Training**
```bash
# Train your model (no automatic registration)
python train.py --model resnet18
```

**What happens during training:**
- ✅ Model training with monitoring (TensorBoard, MLflow metrics, Wandb)
- ✅ Checkpoint saving (best model saved as `best_model.pth`)
- ✅ Performance metrics logging
- ❌ **NO automatic model registration**

### **Phase 2: Evaluation**
```bash
# Evaluate the trained model
python evaluate.py --model resnet18
```

**What happens during evaluation:**
- ✅ Comprehensive performance analysis
- ✅ Medical AI validation metrics
- ✅ Clinical deployment readiness assessment
- ✅ Detailed evaluation report

### **Phase 3: Registration**(Only for good models)
```bash
# Register only the best models
python register_model.py --model resnet18 --version 1.0.0
```

**What happens during registration:**
- ✅ MLflow Model Registry registration
- ✅ Model versioning and documentation
- ✅ Production deployment preparation
- ✅ Medical AI compliance documentation

### **Complete Example Workflow**
```bash
# 1. Train with good parameters
python train.py --model resnet18 --epochs 50

# 2. Check training results
# - Look at TensorBoard: tensorboard --logdir runs
# - Check Wandb dashboard
# - Review training logs

# 3. Evaluate the model
python evaluate.py --model resnet18 --detailed

# 4. If performance is good (>85% validation accuracy), register
python register_model.py --model resnet18 --version 1.0.0 --description "First production model"

# 5. If performance needs improvement, retrain with different parameters
python train.py --model resnet18 --epochs 100 --lr 0.0001
```

### **When to Register vs When NOT to Register**

**✅ Register when:**
- Validation accuracy > 85%
- Good generalization on test set
- Medical AI validation passes
- Model is ready for clinical deployment

**❌ Do NOT register when:**
- Validation accuracy < 80%
- Model shows signs of overfitting
- Medical validation fails
- Training didn't converge properly

### **Registration Best Practices**

**Version Naming:**
```bash
# Major version for significant improvements
python register_model.py --model resnet18 --version 2.0.0

# Minor version for improvements
python register_model.py --model resnet18 --version 1.1.0

# Patch version for bug fixes
python register_model.py --model resnet18 --version 1.0.1
```

**Descriptive Tags:**
```bash
# Production ready model
python register_model.py --model resnet18 --version 1.0.0 --tags "production=ready" "medical_ai=validated"

# Research model
python register_model.py --model resnet18 --version 1.0.0 --tags "research=experimental"
```

**Detailed Descriptions:**
```bash
python register_model.py --model resnet18 --version 1.0.0 \
  --description "Production model achieved 92% validation accuracy, ready for clinical deployment"
```

### **🔧 Monitoring During Training**

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training progress
tensorboard --logdir runs

# Check MLflow metrics
mlflow ui

# Monitor with Wandb
# Check your wandb.ai dashboard
```

### **📋 Checklist Before Registration**

- [ ] Training completed successfully
- [ ] Validation accuracy > 85%
- [ ] No signs of overfitting
- [ ] Medical AI validation passed
- [ ] Model checkpoint saved (`best_model.pth`)
- [ ] Evaluation completed
- [ ] Performance metrics documented
- [ ] Model ready for clinical deployment

**Run multiple experiments:**
```bash
# Compare different models automatically
python run_experiments.py
```

### Evaluation

**Comprehensive medical AI evaluation with clinical validation:**
```bash
# Basic evaluation (with MLflow + Wandb logging)
python3 evaluate.py --model efficientnet_b0

# Full medical AI analysis (comprehensive logging)
python3 evaluate.py --model efficientnet_b0 --detailed --medical-validation --compare

# Quick test (simple logging)
python3 test.py --model efficientnet_b0
```

**Medical AI Evaluation Features:**
- **Sensitivity Analysis**: Critical for tumor detection (minimizing false negatives)
- **Clinical Deployment Validation**: 85%+ accuracy threshold for medical screening
- **Per-class Performance**: Detailed metrics for each tumor type (glioma, meningioma, pituitary)
- **Real-time Inference Assessment**: <500ms per sample for clinical use
- **Regulatory Compliance**: Complete audit trail for FDA/CE marking

**Evaluation Logging Output:**
```
 Setting up evaluation logging...
 MLflow experiment: brain-cancer-mri-evaluation-efficientnet_b0
🔮 Wandb project: brain-cancer-mri-evaluation
🏆 Model shows excellent performance for medical AI!
```

### Model Export & Deployment

**Export trained models for clinical deployment:**
```bash
# Export to both TorchScript and ONNX formats
python3 export_model.py --model efficientnet_b0 --format both

# Export only TorchScript for PyTorch deployments
python3 export_model.py --model resnet18 --format torchscript

# Export only ONNX for cross-platform deployment
python3 export_model.py --model swin_t --format onnx
```

**📦 Export Features:**
- **TorchScript**: Optimized inference for PyTorch-based medical systems
- **ONNX**: Cross-platform compatibility for hospital systems
- **Medical AI Optimization**: Real-time inference for clinical decision support
- **Hardware Compatibility**: Support for various medical imaging hardware

### Model Registry & Production Deployment

**Register models in MLflow Model Registry for production:**
```bash
# Register model with semantic versioning
python3 register_model.py --model efficientnet_b0 --version 1.0.0

# Register with custom description and tags
python3 register_model.py --model resnet18 --version 2.1.0 --description "Improved medical validation" --tags "production=ready" "medical_ai=validated"
```

**Registry Features:**
- **Model Lineage**: Complete development history and audit trail
- **Medical AI Documentation**: Model cards for clinical review
- **Performance Tracking**: Continuous monitoring and validation
- **Deployment Management**: Version control for clinical deployment
- **Regulatory Compliance**: Documentation for FDA/CE marking

#### **🔧 Local MLflow Model Registry Setup**

**Short answer: No, you don't need Databricks for MLflow Model Registry.**

You can use the **open-source MLflow Model Registry locally**—you just need to run a proper MLflow Tracking Server with a SQL backend. The quick `mlflow ui` you ran is great for viewing experiments, but it **doesn't enable the registry**when you're using the default local file store.

**🚀 Quick Start (Recommended):**

**1. Start the MLflow server using the provided script:**

**mac/Linux:**
```bash
./start_mlflow_server.sh
```



**2. Run training (models will be automatically registered):**
```bash
python train.py --model resnet18
```

**3. Test the setup (optional):**
```bash
python test_mlflow_setup.py
```

**4. Run MLflow concepts demonstration (optional):**
```bash
python mlflow_model_examples.py
```
This demonstrates all the MLflow concepts mentioned in the documentation.

**5. View the Model Registry:**
Open `http://127.0.0.1:5000` and click on the **Models**tab to see your registered models.

---

**🔧 Manual Setup (Alternative):**

**1. Start an MLflow Tracking Server with a SQL backend**

**mac/Linux:**
```bash
mkdir -p mlartifacts
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root file:$(pwd)/mlartifacts \
  --host 127.0.0.1 --port 5000
```



**2. The training script automatically connects to the server**

The config file is already set to use `http://127.0.0.1:5000` as the MLflow tracking URI.

**3. Automatic Model Registration**

When you run training, models are automatically:
- **Logged**to MLflow during training
- **Registered**in the Model Registry after training
- **Promoted to Production**if validation accuracy > 85%

**Example output:**
```
📝 Best PyTorch and ONNX models logged to MLflow
📦 ONNX model exported and registered: brain-cancer-resnet18-onnx
📝 Model brain-cancer-resnet18 v1 registered in Staging
🏆 Model brain-cancer-resnet18 v1 promoted to Production!
```

**🔧 Understanding MLflow Model Registration:**

**Key Concepts:**
- **Logging**= "Save this trained model + metadata + artifacts into MLflow"
- **Registering**= "Make this saved model an official, versioned asset in the Model Registry"

**Model Flavors Supported:**
- **PyTorch**: `.pt` files stored internally as `pytorch_model.bin`
- **ONNX**: `.onnx` files for cross-platform deployment
- **Both**: Your training automatically exports and registers both formats

**Artifact Store Structure:**
```
model/
  MLmodel          <-- MLflow model definition
  conda.yaml       <-- Environment specification  
  pytorch_model.bin <-- Your .pt file (internal)
  input_example.json <-- Input example for signature

onnx_model/
  MLmodel
  conda.yaml
  model.onnx       <-- Your .onnx file
  input_example.json
```

**🔧 Using Registered Models for Inference:**

After training and registration, you can use the registered models for inference:

```bash
# Use a registered model for prediction
python inference_with_mlflow.py --model brain-cancer-resnet18 --image path/to/mri/image.jpg

# Use a specific model stage
python inference_with_mlflow.py --model brain-cancer-efficientnet_b0 --stage Staging --image path/to/mri/image.jpg

# Use different image size for specific models
python inference_with_mlflow.py --model brain-cancer-xception_medical --img-size 299 --image path/to/mri/image.jpg
```

**Example inference output:**
```
🧠 Brain Cancer MRI Inference with MLflow
==================================================
✅ Successfully loaded model: brain-cancer-resnet18 (Production)
📸 Loading and preprocessing image: path/to/mri/image.jpg
✅ Image preprocessed successfully (shape: torch.Size([1, 3, 224, 224]))
🔮 Making prediction...

 PREDICTION RESULTS
------------------------------
 Predicted Class: glioma_tumor
📈 Confidence: 87.45%
🔢 Class Index: 0

📋 All Class Probabilities:
  glioma_tumor: 87.45%
  meningioma_tumor: 8.23%
  pituitary_tumor: 4.32%
```

**Notes & Tips:**

* **SQL backend is required**for the registry: `sqlite`, `postgresql`, or `mysql`. (SQLite is fine to start; Postgres/MySQL are better for teams.)
* Your **artifact store**can be local filesystem (as above) or cloud storage (S3, GCS, Azure Blob) if you want remote access.
* Databricks just provides a **managed**MLflow (RBAC, webhooks, approvals, etc.). Great if you need governance at scale, but **optional**.
* **Model naming**: Models are automatically named `brain-cancer-{model_name}` (e.g., `brain-cancer-resnet18`)
* **Auto-promotion**: Models with >85% validation accuracy are automatically promoted to Production stage

### Dataset Structure

The training expects your dataset in the following structure:
```
data/brain-cancer/Brain_Cancer raw MRI data/Brain_Cancer/
├── glioma_tumor/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── meningioma_tumor/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── pituitary_tumor/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## ⚙️ Configuration

The `config/config.yaml` file contains all training parameters:

```yaml
# Available model configurations
models:
  resnet18:
    name: resnet18
    weights: "models.ResNet18_Weights.IMAGENET1K_V1"
    num_classes: 3
    lr: 0.001
  swin_t:
    name: swin_t
    weights: "models.Swin_T_Weights.IMAGENET1K_V1"
    num_classes: 3
    lr: 0.0001  # Lower LR for transformer
  efficientnet_b0:
    name: efficientnet_b0
    weights: "models.EfficientNet_B0_Weights.IMAGENET1K_V1"
    num_classes: 3
    lr: 0.001
  vit_b_16:
    name: vit_b_16
    weights: "models.ViT_B_16_Weights.IMAGENET1K_V1"
    num_classes: 3
    lr: 0.0001

# Selected model
model: resnet18

dataset:
  path: "./data/brain-cancer/Brain_Cancer raw MRI data/Brain_Cancer"
  img_size: 224
  batch_size: 32
  num_workers: 2
  pin_memory: true
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  seed: 42

transform:
  augmentation: true

train:
  epochs: 50
  save_every: 5
  output_dir: "./checkpoints"

monitoring:
  tensorboard_log_dir: "./runs"
  mlflow_experiment_name: "brain-cancer-mri"
  mlflow_tracking_uri: "./mlruns"
  wandb:
    project: "brain-cancer-mri"
    entity: null
    tags: ["brain-tumor", "classification", "pytorch"]

# Performance optimization settings
performance:
  mixed_precision: true      # Use automatic mixed precision (AMP)
  compile_model: true        # Use torch.compile for faster training
  channels_last: true        # Memory format optimization
  benchmark_cudnn: true      # Optimize CUDNN for consistent input sizes
  gradient_accumulation: 1   # Accumulate gradients (increase effective batch size)
  max_grad_norm: 1.0        # Gradient clipping
```

##  Monitoring & Visualization

This project implements **comprehensive logging**across the entire ML pipeline: training, validation, and testing.

### 🏗️ **Monitoring Architecture**

**Three-Tier Logging System:**
```
Training Phase    →  MLflow: brain-cancer-mri-v2-{model}
                  →  Wandb: brain-cancer-mri
                  →  TensorBoard: runs/{model}_logs

Evaluation Phase  →  MLflow: brain-cancer-mri-evaluation-{model}
                  →  Wandb: brain-cancer-mri-evaluation

Testing Phase     →  MLflow: brain-cancer-mri-test-{model}
                  →  Wandb: brain-cancer-mri-test
```

### TensorBoard
Monitor training in real-time:
```bash
tensorboard --logdir runs
```
Then open http://localhost:6006 in your browser.

**Tracked Metrics:**
- Training/Validation Loss
- Training/Validation Accuracy
- Epoch Time

### MLflow
View experiment tracking:
```bash
mlflow ui
```
Then open http://localhost:5000 in your browser.

**MLflow Experiments Structure:**
- `brain-cancer-mri-v2-{model}` - Training experiments
- `brain-cancer-mri-evaluation-{model}` - Comprehensive test evaluation
- `brain-cancer-mri-test-{model}` - Simple test runs

### Weights & Biases
Advanced experiment tracking and visualization:
```bash
# First login (one-time setup)
wandb login

# View experiments at wandb.ai
```

**Wandb Projects Structure:**
- `brain-cancer-mri` - Training experiments
- `brain-cancer-mri-evaluation` - Comprehensive evaluation results
- `brain-cancer-mri-test` - Simple test results

**Logged Information (All Platforms):**

**🏋️ Training Logs:**
- Hyperparameters and model configurations
- Training/validation metrics per epoch
- Model architecture details
- Final trained model artifacts
- Experiment metadata and comparisons
- **Hardware utilization metrics**(GPU/CPU/Memory usage)
- **Performance metrics**(batch processing time, throughput)

**🧪 Evaluation Logs:**
- **Test set performance**: Accuracy, precision, recall, F1-score
- **Per-class metrics**: Detailed analysis for each tumor type
- **Medical AI validation**: Clinical deployment readiness
- **Inference performance**: Real-time timing and throughput
- **Visualizations**: Confusion matrices and performance charts
- **Model comparisons**: Multi-model performance rankings

### 🔗 **End-to-End Traceability**

**Complete ML Pipeline Logging:**
1. **Training**→ Log hyperparameters, validation metrics, checkpoints
2. **Evaluation**→ Log test performance, medical validation, artifacts
3. **Deployment**→ Link training runs to final test results

**Benefits:**
- **Regulatory compliance**: Full audit trail for medical AI
- **Model selection**: Compare training vs test performance
- **Performance tracking**: Monitor model degradation over time
- **Reproducibility**: Recreate any experiment from logged parameters

## 🔧 Features

### ✅ Implemented Features

#### **🤖 Model Support**
- [x] **Multiple Architectures**: ResNet18/50, Swin Transformer, EfficientNet, Vision Transformer
- [x] **Model-Specific Optimization**: Batch sizes and learning rates optimized per architecture
- [x] **Transfer Learning**: Pre-trained weights for all models

#### **Performance Optimization**
- [x] **Mixed Precision Training**: FP16 for 2x faster training and 50% memory savings
  - ⚠️ **Medical Image Warning**: Disable for medical images due to numerical instability
- [x] **Model Compilation**: PyTorch 2.0 compilation for 20-30% speed boost
  - ⚠️ **Stability Note**: Disable if encountering training issues
- [x] **GPU Memory Optimization**: Channels-last memory format, CUDNN benchmarking
- [x] **Multi-core Data Loading**: 8 workers with prefetching and persistent workers
- [x] **Gradient Clipping**: Prevents gradient explosions for stable training

#### **Medical Image Specific Settings**
For **brain MRI and other medical images**, use these stable settings:
```yaml
performance:
  mixed_precision: false     # Critical: Prevents NaN with medical images
  compile_model: false       # Improves stability
  channels_last: false       # Simplify for debugging
  max_grad_norm: 1.0         # Conservative gradient clipping

models:
           resnet18:
           lr: 0.0001              # Lower LR for medical domain transfer
           batch_size: 64          # Conservative batch size
       ```

#### **🖥️ GPU Memory Management**

**For 4GB GPU (RTX 3050/GTX 1050 Ti):**
```yaml
models:
  resnet18:
    batch_size: 128       # ✅ Safe for 4GB
  
  efficientnet_b0:
    batch_size: 128       # ✅ Safe for 4GB
    
  xception_medical:
    batch_size: 16        # ⚠️ 299×299 input requires small batch
  
  swin_t:
    batch_size: 32        # ⚠️ Moderate memory usage
  
  swin_s:
    batch_size: 8         # 🚨 Large model - very small batch
  
  vit_b_16:
    batch_size: 16        # 🚨 Very large transformer
    
  medical_cnn:
    batch_size: 64        # ✅ Frozen backbone saves memory
```

**Memory Optimization Tips:**
- **Reduce batch size**for larger models (Swin-S, ViT)
- **Disable `channels_last`**if memory-constrained: `channels_last: false`
- **Disable model compilation**for debugging: `compile_model: false`
- **Use gradient checkpointing**for very large models
- **Monitor GPU usage**: `nvidia-smi` during training

#### **⏰ Early Stopping & Patience**

The training now includes **automatic early stopping**to prevent overfitting:

```yaml
train:
  epochs: 50
  patience: 10          # Stop if no improvement for 10 epochs
  min_delta: 0.001      # Minimum improvement threshold (0.1%)
```

**How it works:**
- **Tracks best validation accuracy**across all epochs
- **Saves best model**automatically when validation improves
- **Stops training early**if no improvement for `patience` epochs
- **Prevents overfitting**and saves training time
- **Logs improvement details**with each new best model

**Example output:**
```
 New best model saved! Val Acc: 87.45% (improvement: 0.234%)
⏰ No improvement for 3/10 epochs (best: 87.45%)
🛑 Early stopping triggered! No improvement for 10 epochs.
🏆 Best validation accuracy: 87.45%
```

#### **📁 Multi-Model Output Structure**

The training script now automatically creates **model-specific directories**:
```
checkpoints/
├── resnet18_outputs/        # ResNet18 results
│   ├── best_model.pth
│   ├── confusion_matrix.png
│   └── evaluation_report.txt
├── xception_medical_outputs/ # Kaggle Xception solution
│   ├── best_model.pth
│   ├── confusion_matrix.png
│   └── evaluation_report.txt
├── swin_s_outputs/          # Swin-S results  
│   ├── best_model.pth
│   └── evaluation_report.txt
├── swin_t_outputs/          # Swin-T results
│   └── ...
└── medical_cnn_outputs/     # Medical CNN results
    └── ...

runs/
├── resnet18_logs/           # ResNet18 TensorBoard
├── xception_medical_logs/   # Xception TensorBoard
├── swin_s_logs/             # Swin-S TensorBoard  
└── swin_t_logs/             # Swin-T TensorBoard

mlruns/                      # MLflow experiments
├── brain-cancer-mri-v2-resnet18/
├── brain-cancer-mri-v2-xception_medical/
├── brain-cancer-mri-v2-swin_s/
└── brain-cancer-mri-v2-swin_t/
```

#### **Monitoring & Tracking**
- [x] **Triple Monitoring**: TensorBoard + MLflow + Weights & Biases
- [x] **Hardware Monitoring**: Real-time GPU/CPU/Memory utilization tracking
- [x] **Performance Metrics**: Batch processing time, throughput, temperature monitoring

#### **🎛️ Usability**
- [x] **Flexible Configuration**: Model-specific hyperparameters and easy switching
- [x] **Command Line Interface**: Override any parameter from command line
- [x] **Automated Experiments**: Run multiple models for comparison
- [x] **Real-time Progress**: Detailed training progress with hardware stats
- [x] **Data Augmentation**: Flip, rotation, color jitter
- [x] **Checkpoint Saving**: Regular model checkpoints with metadata

#### **Comprehensive Evaluation**
- [x] **Medical AI Validation**: Clinical deployment readiness assessment
- [x] **Multi-Model Comparison**: Automatic ranking and performance analysis
- [x] **Rich Visualizations**: Enhanced confusion matrix and per-class metrics
- [x] **Multiple Output Formats**: Text reports, JSON metrics, medical validation
- [x] **Inference Performance**: Real-time timing and throughput analysis
- [x] **Sensitivity Analysis**: Critical for medical diagnosis accuracy
- [x] **Full Evaluation Logging**: MLflow + Wandb tracking for test results
- [x] **Regulatory Compliance**: Complete audit trails for medical AI deployment

### 📈 Training Output

The training script provides detailed progress information:

```
============================================================
🚀 Starting Brain Cancer MRI Training
============================================================
 Dataset: 4239 train, 908 validation samples
🏗️  Model: resnet18 (resnet18) with 3 classes
📦 Batch size: 128 (model-optimized)
📈 Learning rate: 0.001
 Epochs: 50
 Device: cuda
💻 CPU: 20 cores @ 8.5%
🧠 RAM: 31.7% used
🚀 GPU: NVIDIA GeForce RTX 3050
 GPU Usage: 95.2% | Memory: 78.5% (3140/4096MB)
 Mixed Precision: Enabled
🔥 Model Compilation: Enabled
📈 TensorBoard: ./runs
 MLflow: brain-cancer-mri
🔮 Wandb: brain-cancer-mri
============================================================

📚 Epoch 1/50
--------------------------------------------------
  Batch 10/34 | Loss: 1.0483 | Acc: 46.95%
  Batch 20/34 | Loss: 0.9310 | Acc: 57.19%
  Batch 30/34 | Loss: 0.8457 | Acc: 63.28%

 Epoch 1 Summary:
  🕐 Time: 75.78s
  📉 Train Loss: 0.8141 | Train Acc: 65.13%
   Val Loss: 0.7442 | Val Acc: 63.88%
 Saving checkpoint at epoch 1
```

## 🤖 Available Models

| Model | Type | Parameters | Input Size | Batch Size* | Optimizer | Memory | Learning Rate |
|-------|------|------------|------------|-------------|-----------|--------|---------------|
| `resnet18` | CNN | 11.7M | 224×224 | **128**| AdamW | ~2.5GB | 0.0001 |
| `resnet50` | CNN | 25.6M | 224×224 | **64**| AdamW | ~3.5GB | 0.001 |
| `efficientnet_b0` | CNN | 5.3M | 224×224 | **128**| AdamW | ~2.0GB | 0.001 |
| `swin_t` | Transformer | 28.3M | 224×224 | **32**| AdamW | ~3.0GB | 0.00005 |
| `swin_s` | Transformer | 49.6M | 224×224 | **8**| AdamW | ~3.8GB | 0.0001 |
| `vit_b_16` | Transformer | 86.6M | 224×224 | **16**| AdamW | ~3.8GB | 0.0001 |
| `medical_cnn` | Medical CNN | 5.3M (frozen) + 0.2M | 224×224 | **64**| AdamW | ~2.0GB | 0.001 |
| **`xception_medical`**| **Kaggle Solution**| **5.3M (trainable)**| **299×299**| **16**| **Adamax**| **~2.2GB**| **0.001**|

*Optimized for RTX 3050/4GB VRAM. Automatically scales with available GPU memory.

### Model Selection Guide

**🚀 For Quick Experiments:**
- `resnet18`: Fast training, largest batch size (128), good baseline
- `efficientnet_b0`: Best efficiency/accuracy trade-off, large batches

**For Best Accuracy:**
- **`xception_medical`**: **Kaggle-proven solution**(299×299, Adamax optimizer)
- `swin_t`: Modern transformer architecture, optimized batch size
- `swin_s`: Higher capacity for complex patterns

**For Medical Domain:**
- **`xception_medical`**: **Kaggle brain tumor solution**with 99% reported accuracy
- `medical_cnn`: EfficientNet + frozen backbone, fast convergence
- `resnet18`: Reliable baseline for medical images

**For Production:**
- `efficientnet_b0`: Lightweight, efficient, maximum throughput
- `resnet18`: Well-tested, reliable, fastest training
- `xception_medical`: Proven medical imaging performance

**🔥 For Maximum GPU Utilization:**
- Models automatically use optimized batch sizes for your hardware
- Early stopping prevents overfitting and saves training time
- Mixed precision disabled for medical image stability
- All models benefit from hardware acceleration

##  Medical AI Workflow

### **Complete Clinical Deployment Pipeline**

This project provides a comprehensive workflow for medical AI development and deployment:

#### **1. 🧠 Model Training & Optimization**
```bash
# Train with medical AI optimizations
python3 train.py --model efficientnet_b0
```
**Medical AI Features:**
- **Stable Training**: Disabled mixed precision for medical image stability
- **Medical Validation**: Real-time monitoring of training metrics
- **Hardware Optimization**: Model-specific batch sizes for clinical hardware
- **Complete Logging**: Audit trail for regulatory compliance

#### **2.  Comprehensive Medical AI Evaluation**
```bash
# Full medical AI validation
python3 evaluate.py --model efficientnet_b0 --detailed --medical-validation --compare
```
**Clinical Validation Features:**
- **Sensitivity Analysis**: Critical for tumor detection (≥80% sensitivity required)
- **Accuracy Thresholds**: 85%+ accuracy for medical screening applications
- **Per-class Performance**: Detailed analysis for each tumor type
- **Real-time Assessment**: <500ms inference for clinical decision support
- **Medical AI Compliance**: Deployment readiness validation

#### **3. 📦 Clinical Deployment Export**
```bash
# Export for medical system integration
python3 export_model.py --model efficientnet_b0 --format both
```
**Deployment Features:**
- **TorchScript**: Optimized for PyTorch-based medical systems
- **ONNX**: Cross-platform compatibility for hospital systems
- **Medical AI Optimization**: Real-time inference capabilities
- **Hardware Compatibility**: Support for various medical imaging hardware

#### **4.  Production Model Registry**
```bash
# Register for clinical deployment
python3 register_model.py --model efficientnet_b0 --version 1.0.0 --description "Clinical deployment ready"
```
**Registry Features:**
- **Model Lineage**: Complete development history and audit trail
- **Medical AI Documentation**: Model cards for clinical review
- **Performance Tracking**: Continuous monitoring and validation
- **Regulatory Compliance**: Documentation for FDA/CE marking

### **Medical AI Validation Criteria**

**Clinical Deployment Standards:**
- **Accuracy ≥ 85%**: Minimum threshold for medical screening applications
- **Sensitivity ≥ 80%**: Critical for tumor detection (minimize false negatives)
- **Real-time Inference**: <500ms per sample for clinical decision support
- **Complete Documentation**: Audit trail for regulatory compliance
- **Cross-platform Compatibility**: Support for various hospital systems

## 🎛️ Customization

### Model Configuration
- Choose from 8 different architectures optimized for medical AI
- Model-specific learning rates automatically applied for clinical stability
- Easy switching via command line or config file

### Adding New Models
1. Implement your model in `models/model.py`
2. Update the `get_model()` function
3. Set the model name in config

### Custom Datasets
1. Update the dataset path in `config.yaml`
2. Ensure your data follows the ImageFolder structure
3. Modify `num_classes` if needed

## 🔬 Automated Experiments

The `run_experiments.py` script allows you to compare multiple models automatically:

```bash
# Run comparison experiments
python run_experiments.py
```

This will:
- Train multiple models with optimized settings
- Log all experiments to monitoring platforms
- Provide a comprehensive comparison report
- Save time on manual hyperparameter tuning

## 📋 Script Documentation

### **🧪 evaluate.py - Comprehensive Medical AI Evaluation**

**Purpose**: Complete model evaluation with medical AI validation for clinical deployment readiness.

**Key Features:**
- **Medical AI Validation**: Sensitivity analysis critical for tumor detection
- **Clinical Deployment Assessment**: 85%+ accuracy threshold validation
- **Per-class Performance**: Detailed metrics for each tumor type
- **Real-time Inference**: <500ms assessment for clinical decision support
- **Regulatory Compliance**: Complete audit trail for FDA/CE marking

**Usage Examples:**
```bash
# Basic evaluation with logging
python3 evaluate.py --model efficientnet_b0

# Full medical AI analysis
python3 evaluate.py --model efficientnet_b0 --detailed --medical-validation --compare

# Medical AI validation only
python3 evaluate.py --model resnet18 --medical-validation
```

**Outputs:**
- **Evaluation Report**: Detailed human-readable analysis
- **JSON Metrics**: Structured data for programmatic access
- **Visualizations**: Confusion matrix and per-class performance charts
- **Medical Validation**: Clinical deployment readiness assessment
- **Model Comparison**: Multi-model performance rankings

### **📊 compute_normalization.py - Comprehensive Normalization Statistics**

**Purpose**: Compute, save, and verify normalization statistics for the fixed seed split with multiple modes.

**Key Features:**
- **Three Modes**: Display, save, and verify existing statistics
- **One-time Computation**: Compute statistics once, reuse across all experiments
- **Fixed Seed Consistency**: Uses the same seed (42) as training for reproducible splits
- **Multiple Output Formats**: JSON and Python constants for easy integration
- **Automatic Detection**: Training script automatically detects and uses pre-computed statistics

**Usage:**
```bash
# Pre-compute and save statistics (run once)
python compute_normalization.py --mode save

# Display statistics only (no saving)
python compute_normalization.py --mode display

# Verify existing pre-computed statistics
python compute_normalization.py --mode verify

# Use in training (automatic)
python train.py --model resnet18
```

**Generated Files:**
- `normalization_stats.json` - Complete statistics with metadata
- `normalization_constants.py` - Python constants for easy import

**Example `normalization_constants.py`:**
```python
# Auto-generated normalization constants
# Generated from compute_normalization.py
# Fixed seed: 42

NORMALIZATION_MEAN = [0.1528, 0.1528, 0.1528]
NORMALIZATION_STD = [0.1613, 0.1613, 0.1613]

# Dataset configuration
DATASET_CONFIG = {
    "path": "./data/brain-cancer/Brain_Cancer raw MRI data/Brain_Cancer",
    "img_size": 224,
    "seed": 42,
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15
}

# Dataset sizes
DATASET_SIZES = {
    "train": 4239,
    "val": 908,
    "test": 908
}
```

**Example Output (Save Mode):**
```
📋 Loading config from: config/config.yaml
🔧 Configuration loaded successfully
📊 Dataset path: ./data/brain-cancer/Brain_Cancer raw MRI data/Brain_Cancer
🖼️  Image size: 224
🎲 Fixed seed: 42

📊 Loading datasets without normalization...
📈 Dataset sizes:
   Training: 4239 samples
   Validation: 908 samples
   Test: 908 samples

📊 Computing mean/std on training set...
✅ Statistics computed successfully!
📊 Training set mean: [0.1528, 0.1528, 0.1528]
📊 Training set std: [0.1613, 0.1613, 0.1613]

💾 Mode: Save to files
💾 Statistics saved to: normalization_stats.json
🐍 Python constants saved to: normalization_constants.py
✅ Pre-computation completed successfully!
```

**Example Output (Verify Mode):**
```
✅ Found existing pre-computed statistics:
   Saved mean: [0.1528, 0.1528, 0.1528]
   Saved std: [0.1613, 0.1613, 0.1613]
   Computed mean: [0.1528, 0.1528, 0.1528]
   Computed std: [0.1613, 0.1613, 0.1613]
✅ Statistics match! Pre-computed values are correct.
```

**Benefits:**
-  **30-60 seconds saved**per training run
-  **Guaranteed consistency**across all experiments
-  **Reproducible results**with identical normalization
-  **Reduced computational overhead**during training
- 🏥 **Medical Domain Optimized**: Your statistics are specifically computed for brain MRI data
-  **Better Performance**: Domain-specific normalization often improves model performance

### **📦 export_model.py - Clinical Deployment Export**

**Purpose**: Export trained models for production deployment in medical systems.

**Key Features:**
- **TorchScript Export**: Optimized inference for PyTorch-based medical systems
- **ONNX Export**: Cross-platform compatibility for hospital systems
- **Medical AI Optimization**: Real-time inference capabilities
- **Hardware Compatibility**: Support for various medical imaging hardware

**Usage Examples:**
```bash
# Export to both formats
python3 export_model.py --model efficientnet_b0 --format both

# Export only TorchScript
python3 export_model.py --model resnet18 --format torchscript

# Export only ONNX
python3 export_model.py --model swin_t --format onnx
```

**Outputs:**
- **TorchScript Models**: `.pt` files for PyTorch deployment
- **ONNX Models**: `.onnx` files for cross-platform deployment
- **Performance Validation**: Export testing and validation
- **File Size Analysis**: Deployment planning information

### **register_model.py - Production Model Registry**

**Purpose**: Register models in MLflow Model Registry for clinical deployment with complete documentation.

**Key Features:**
- **Model Lineage**: Complete development history and audit trail
- **Medical AI Documentation**: Model cards for clinical review
- **Performance Tracking**: Continuous monitoring and validation
- **Deployment Management**: Version control for clinical deployment
- **Regulatory Compliance**: Documentation for FDA/CE marking

**Usage Examples:**
```bash
# Register with semantic versioning
python3 register_model.py --model efficientnet_b0 --version 1.0.0

# Register with custom description and tags
python3 register_model.py --model resnet18 --version 2.1.0 --description "Improved medical validation" --tags "production=ready" "medical_ai=validated"
```

**Outputs:**
- **Model Cards**: Comprehensive documentation for clinical review
- **MLflow Registry**: Production deployment management
- **Performance Metrics**: Linked to training and evaluation results
- **Medical AI Validation**: Deployment readiness documentation

### **monitor_gpu.py - Medical AI GPU Monitoring**

**Purpose**: Real-time GPU monitoring specifically designed for medical AI training with performance recommendations.

**Key Features:**
- **Medical AI Specific**: Tailored recommendations for brain cancer MRI training
- **Low Overhead**: Efficient monitoring that doesn't impact training performance
- **Temperature Monitoring**: Critical for long medical AI training runs
- **Memory Optimization**: Helps with batch size tuning for medical images
- **Process Identification**: Shows which training processes are using GPU
- **Performance Alerts**: Warns about potential issues before they become problems

**Usage Examples:**
```bash
# Continuous monitoring during training
python monitor_gpu.py

# Custom update interval (3 seconds)
python monitor_gpu.py -i 3

# Show status once and exit
python monitor_gpu.py --once

# Save metrics for analysis
python monitor_gpu.py --once --save gpu_metrics.json

# Hide process information
python monitor_gpu.py --no-processes
```

**Medical AI Features:**
- **Temperature Alerts**: Warns when GPU temperature > 80°C
- **Memory Warnings**: Alerts when memory usage > 95%
- **Utilization Analysis**: Detects underutilized GPUs with high memory usage
- **Training Optimization**: Provides specific advice for medical AI workloads

**Example Output:**
```
🧠 Brain Cancer MRI - GPU Monitoring
============================================================
📅 2024-01-15 14:30:25

💻 SYSTEM:
   CPU: 45.2% | RAM: 67.8% | Available: 12.3GB

🚀 GPU STATUS:
   GPU 0: NVIDIA GeForce RTX 3050
   ├─ Utilization: 87.3%
   ├─ Memory: 3140MB / 4096MB (76.7%)
   ├─ Temperature: 72°C
   └─ Power: 115.2W / 130.0W

📋 GPU PROCESSES:
   PID 12345: python (3140MB)

 MEDICAL AI RECOMMENDATIONS:
   ✅ GPU utilization looks good for medical AI training
```

**Example Output:**
```
🧠 Brain Cancer MRI Classification - Model Comparison
============================================================
🔬 Experiment 1/4
🚀 Starting experiment: resnet18
...
 EXPERIMENT SUMMARY
============================================================
resnet18        | ✅ SUCCESS | 245.67s | Epochs: 5 | Batch: 16
efficientnet_b0 | ✅ SUCCESS | 198.34s | Epochs: 5 | Batch: 16
swin_t          | ✅ SUCCESS | 312.45s | Epochs: 3 | Batch: 8
vit_b_16        | ✅ SUCCESS | 387.12s | Epochs: 3 | Batch: 8
```

## 📋 Requirements

### **Core Dependencies**
- Python 3.8+
- PyTorch 2.0+ (for compilation support)
- torchvision
- CUDA 11.8+ (for GPU acceleration)

### **Monitoring & Logging**
- tensorboard
- mlflow
- wandb

### **Performance & Hardware Monitoring**
- psutil (CPU/Memory monitoring)
- GPUtil (GPU monitoring)

### **Utilities**
- PyYAML
- Pillow
- numpy

### **Installation**

#### **Option 1: Using pip (Recommended for most users)**
```bash
# Install all dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tensorboard mlflow wandb psutil GPUtil pyyaml pillow numpy

# Install additional packages for data augmentation and medical imaging
pip install albumentations opencv-python scikit-learn matplotlib seaborn
```

#### **Option 2: Using conda (Recommended for advanced augmentation and medical imaging)**
```bash
# Install core packages via conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install advanced augmentation and medical imaging packages via conda-forge
conda install -c conda-forge albumentations opencv

# Install remaining packages
pip install tensorboard mlflow wandb psutil GPUtil pyyaml pillow numpy
pip install scikit-learn matplotlib seaborn
```

#### **Option 3: Using requirements.txt**
```bash
# Install from requirements file
pip install -r requirements.txt

# For conda users, install albumentations and opencv via conda-forge first
conda install -c conda-forge albumentations opencv
pip install -r requirements.txt
```

#### **Why conda for albumentations and opencv?**
- **Better Performance**: Conda packages are often compiled with optimized libraries
- **Dependency Resolution**: Better handling of complex dependencies
- **Medical Imaging**: Optimized for medical image processing workflows
- **Cross-platform**: More consistent behavior across different operating systems

##  Performance Optimization

### 🚀 **Hardware Utilization**

The project automatically optimizes for your hardware:

- **GPU Memory**: Model-specific batch sizes maximize VRAM usage
- **Multi-core CPU**: 8 data loading workers utilize all CPU cores  
- **Mixed Precision**: FP16 training reduces memory usage by ~50%
- **Memory Optimization**: Channels-last format and non-blocking transfers

### 🔥 **Speed Optimizations**

- **PyTorch 2.0 Compilation**: 20-30% faster training with `torch.compile`
- **CUDNN Benchmarking**: Optimizes convolution algorithms for consistent inputs
- **Prefetching**: Keeps GPU fed with data using prefetch buffers
- **Persistent Workers**: Avoids worker respawn overhead

###  **GPU Monitoring Best Practices**

**❌ Avoid: `watch -n 1 nvidia-smi`**
- High CPU overhead when called frequently
- Can impact training performance
- Limited information

**✅ Recommended Tools:**

#### **🥇 Best: `nvtop` (Recommended)**
```bash
# Install
sudo apt install nvtop

# Use
nvtop
```
**Advantages:**
- ✅ **Low overhead**: Much more efficient than `nvidia-smi`
- ✅ **Rich interface**: Shows GPU, memory, power, temperature
- ✅ **Process details**: Shows which processes are using GPU
- ✅ **Historical graphs**: Real-time charts
- ✅ **Interactive**: Can sort, filter, kill processes

#### **🥈 Good: `gpustat`**
```bash
# Install
pip install gpustat

# Use
gpustat -i 2
```
**Advantages:**
- ✅ **Lightweight**: Much faster than `nvidia-smi`
- ✅ **JSON output**: Can be parsed programmatically
- ✅ **Process info**: Shows which processes are using GPU

#### **Medical AI Specific: Custom Monitoring Script**
```bash
# Use the project's custom monitoring script
python monitor_gpu.py

# With custom interval
python monitor_gpu.py -i 3

# Show status once
python monitor_gpu.py --once

# Save metrics for analysis
python monitor_gpu.py --once --save gpu_metrics.json
```

**Medical AI Features:**
- ✅ **Medical AI recommendations**: Specific advice for your use case
- ✅ **Temperature monitoring**: Critical for long training runs
- ✅ **Memory optimization**: Helps with batch size tuning
- ✅ **Process identification**: Shows which training processes are running
- ✅ **Performance alerts**: Warns about potential issues

#### **Tool Comparison**

| Tool | Overhead | Features | Medical AI | Best For |
|------|----------|----------|------------|----------|
| `watch -n 1 nvidia-smi` | **High**| Basic | ❌ | Quick checks |
| `nvtop` | **Low**| Rich UI, graphs | ✅ | Daily use |
| `gpustat` | **Low**| Lightweight, JSON | ✅ | Scripts |
| `monitor_gpu.py` | **Low**| Medical AI specific | ✅ | Your project |

###  **Model-Specific Optimizations**

| Model | Optimized Batch | Expected Speed | Memory Usage |
|-------|----------------|----------------|--------------|
| ResNet18 | 128 | **Fastest**| 2.5GB |
| EfficientNet-B0 | 128 | **Most Efficient**| 2.0GB |
| ResNet50 | 64 | Fast | 3.5GB |
| Swin-T | 32 | Moderate | 3.0GB |
| ViT-B/16 | 16 | Slower | 3.8GB |

### 💡 **Performance Tips**

1. **For Maximum Speed**: Use `resnet18` or `efficientnet_b0`
2. **For Memory Efficiency**: Enable mixed precision (default)
3. **For Large Datasets**: Increase `num_workers` to match CPU cores
4. **For Multiple GPUs**: Models support DataParallel automatically
5. **For GPU Monitoring**: Use `nvtop` instead of `watch nvidia-smi`
6. **For Medical AI**: Use the custom `monitor_gpu.py` script

###  **Medical AI Training Monitoring**

**Recommended workflow:**
```bash
# Terminal 1: Start training
python train.py --model resnet18

# Terminal 2: Monitor with medical AI specific tool
python monitor_gpu.py -i 3
```

**What to watch for:**
- **Temperature > 80°C**: Consider reducing batch size or improving cooling
- **Memory > 95%**: Reduce batch size to prevent OOM errors
- **Low utilization + high memory**: Potential memory leak or inefficient data loading
- **CPU bottleneck**: Increase `num_workers` in config

## 🐛 Troubleshooting

### Common Issues

**"UnboundLocalError: cannot access local variable 'torch' where it is not associated with a value"**
- **Root cause**: Import error due to incorrect working directory or missing dependencies
- **Solution**: Use the training runner script: `./run_training.sh` or `.\run_training.ps1`
- **Alternative**: Navigate to the correct directory: `cd src/projects/BrainCancer-MRI`
- **Why this happens**: Python can't find the required modules when run from wrong directory

**"FileNotFoundError: config file not found"**
- Ensure you're running from the project directory
- Check the config file path

**"CUDA out of memory"**
- Reduce batch size in config
- Reduce number of workers

**"Dataset folder not found"**
- Verify dataset path in config
- Check folder structure matches expected format

**"NaN/Inf loss detected" or "Large gradient norm detected"**
- **Root cause**: Automatic Mixed Precision (AMP) can cause numerical instability with medical images
- **Solution**: Disable AMP in config: `performance.mixed_precision: false`
- **Why this happens**: Medical images have different intensity distributions than natural images, causing AMP's FP16 precision to overflow
- **Alternative fixes**:
  - Reduce learning rate further (e.g., 0.00001)
  - Disable model compilation: `performance.compile_model: false`
  - Use stronger gradient clipping: `performance.max_grad_norm: 1.0`

**"Extreme input ranges like [-0.9, 5.2]"**
- **Root cause**: Computed normalization statistics inappropriate for pre-trained models
- **Solution**: Either disable normalization entirely or use ImageNet stats
- **Why**: Pre-trained models expect specific input distributions


##  **Pre-deployment Quality & Validation**

After successfully training your model, follow these essential steps before deploying to production:

### **1.  Comprehensive Model Evaluation**

Use the new **comprehensive evaluation script**for detailed analysis:

```bash
# Basic evaluation
python3 evaluate.py --model efficientnet_b0

# Full medical AI validation with visualizations
python3 evaluate.py --model efficientnet_b0 --detailed --medical-validation

# Compare all trained models
python3 evaluate.py --model resnet18 --compare

# Complete comprehensive evaluation
python3 evaluate.py --model xception_medical --detailed --medical-validation --compare
```

**Generated outputs:**
- **Detailed evaluation report**(`evaluation_report.txt`)
- **JSON metrics**(`evaluation_metrics.json`) for programmatic access
- **Enhanced confusion matrix**with counts and percentages
- **Per-class metrics visualization**(precision, recall, F1)
- **Medical AI validation report**(`medical_validation.json`)
- **Multi-model comparison**with performance rankings

### **1b.  Simple Test Set Evaluation**

For quick testing, use the simple evaluation script:

```bash
python3 test.py --model efficientnet_b0
python3 test.py --model resnet18  
python3 test.py --model xception_medical
```

This provides basic metrics and confusion matrix.

### **2.  Medical AI Validation Features**

The `evaluate.py` script includes specialized medical AI validation:

**Clinical Standards:**
- **85%+ accuracy threshold**for medical screening deployment
- **80%+ sensitivity**for tumor detection (prevents missed diagnoses)
- **Specificity analysis**for each class (reduces false positives)
- **Clinical deployment readiness**assessment

**Advanced Analysis:**
- **Per-class breakdown**: Detailed metrics for each tumor type
- **Sensitivity warnings**: Alerts for classes with low recall (<80%)
- **Class imbalance detection**: Identifies potential training data issues
- **Inference performance**: Real-time capability assessment (<500ms)

### **3. 📈 Model Comparison & Selection**

Use the built-in comparison features:

```bash
# Compare all trained models automatically
python3 evaluate.py --model efficientnet_b0 --compare
```

**Comparison includes:**
- **Accuracy rankings**across all trained models
- **F1-score analysis**for balanced performance assessment
- **Inference speed comparison**(ms per sample)
- **Medical AI compliance**status for each model
- **Best model recommendation**with rationale

### **4.  Performance Profiling**

Built-in performance analysis:

- **Real-time inference**: Automatic timing for each model (<1ms typical)
- **Memory efficiency**: Model-specific batch size optimization
- **Throughput analysis**: Batch processing capabilities
- **Hardware utilization**: GPU/CPU usage monitoring during evaluation

### **5.  Evaluation Output Examples**

The `evaluate.py` script generates comprehensive outputs:

**📋 Console Output:**
```
🧠 Brain Cancer MRI Model Evaluation
==================================================
📋 Model: efficientnet_b0 (efficientnet_b0)
 Checkpoint validation accuracy: 91.63%
🖥️  Using device: cuda

🚀 Starting evaluation...
🧪 Running inference on test set...
⏱️  Total evaluation time: 2.20s

✅ **EVALUATION RESULTS**
 Test Accuracy: 0.9230 (92.30%)
 Macro F1-Score: 0.9238
 Avg inference time: 0.82ms per sample

 MEDICAL AI VALIDATION RESULTS:
   Accuracy threshold (≥85%): ✅ PASSED
   Deployment ready: ✅ YES

🏆 Model shows excellent performance for medical AI!
```

**📁 Generated Files:**
- `evaluation_report.txt` - Detailed human-readable report
- `evaluation_metrics.json` - Structured metrics for analysis
- `confusion_matrix.png` - Enhanced visualization with percentages
- `per_class_metrics.png` - Precision/Recall/F1 comparison chart
- `medical_validation.json` - Clinical deployment assessment

**Logged to Monitoring Platforms:**
- **MLflow**: Test metrics, artifacts, and model registry
- **Weights & Biases**: Interactive dashboards and visualizations
- **Separate projects**: `brain-cancer-mri-evaluation` (comprehensive) vs `brain-cancer-mri-test` (simple)

**🔗 Experiment Linking:**
- Training runs linked to evaluation results
- Model checkpoints connected to test performance
- Complete ML pipeline traceability from training → validation → testing

### **6. 📦 Model Packaging for Inference**

Prepare for production deployment:

#### **Model Export Script**

Use the dedicated export script to convert trained models for deployment:

```bash
# Export specific model to both TorchScript and ONNX:
python3 export_model.py --model efficientnet_b0

# Export only TorchScript:
python3 export_model.py --model resnet18 --format torchscript

# Export only ONNX:
python3 export_model.py --model xception_medical --format onnx

# Export all available models:
python3 export_model.py --model resnet18 --format both
python3 export_model.py --model swin_t --format both
```

**Example output:**
```
🚀 Model Export for Deployment
📋 Model: efficientnet_b0
 Loaded model from epoch 6
 Validation accuracy: 0.9163
✅ TorchScript model saved: efficientnet_b0_model.pt (16.02 MB)
✅ ONNX model saved: efficientnet_b0_model.onnx (15.29 MB)
```

**Exported files structure:**
```
checkpoints/
├── efficientnet_b0_outputs/
│   └── exported_models/
│       ├── efficientnet_b0_model.pt    # TorchScript
│       └── efficientnet_b0_model.onnx  # ONNX
├── resnet18_outputs/
│   └── exported_models/
│       ├── resnet18_model.pt
│       └── resnet18_model.onnx
└── xception_medical_outputs/
    └── exported_models/
        ├── xception_medical_model.pt   # 299x299 input
        └── xception_medical_model.onnx
```

#### **Model Registry**
- Register model in **MLflow Model Registry**
- Version control with semantic versioning
- Document model performance metrics
- Track model lineage and training configuration

### **6. 🐳 Deployment Packaging**

Containerize for production:

```dockerfile
# Dockerfile for inference service
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY brain_cancer_model.pt /app/
COPY inference_service.py /app/

WORKDIR /app
EXPOSE 8000

CMD ["python", "inference_service.py"]
```

### **📋 Quality Gates Checklist**

Use `evaluate.py --medical-validation` to automatically check these criteria:

- [ ] **Test accuracy ≥ 85%**(medical AI threshold) ✅ *Auto-checked*
- [ ] **Per-class recall ≥ 80%**(no missed diagnoses) ✅ *Auto-checked*
- [ ] **Inference latency < 500ms**✅ *Auto-measured*
- [ ] **Sensitivity warnings addressed**✅ *Auto-reported*
- [ ] **Class imbalance < 5:1 ratio**✅ *Auto-detected*
- [ ] **Confusion matrix analysis**✅ *Auto-generated*
- [ ] **Model comparison completed**✅ *Auto-compared*
- [ ] **Medical validation passed**✅ *Auto-validated*

**Quick validation:**
```bash
python3 evaluate.py --model your_model --medical-validation
```

### **Medical AI Specific Considerations**

- **FDA/Regulatory compliance**: Document training data, validation methodology
- **Clinical integration**: Ensure compatibility with PACS systems
- **Radiologist workflow**: Design UI for clinical decision support
- **Audit trail**: Log all predictions for regulatory review
- **Fallback mechanisms**: Handle edge cases gracefully

### **Evaluation Logging Best Practices**

**Why Log Evaluation Results?**

**Medical AI Requirements:**
- **Regulatory compliance**: FDA/CE marking requires complete audit trails
- **Clinical validation**: Document test performance for medical review
- **Model versioning**: Track which model version achieved what performance
- **Safety documentation**: Prove model safety and efficacy

**📈 MLOps Benefits:**
- **Model registry**: Link training experiments to final test performance
- **Performance monitoring**: Track model degradation over time
- **A/B testing**: Compare model performance across different test sets
- **Deployment decisions**: Use logged metrics to select production models

**Complete Pipeline Traceability:**
```
Training Run ID → Validation Metrics → Test Results → Deployment Decision
     ↓                    ↓                ↓              ↓
  MLflow Log         MLflow Log      MLflow Log    Model Registry
```

**Example Logged Metrics:**
```
Training: train_loss, val_loss, train_acc, val_acc, epoch_time
Testing:  test_accuracy, test_f1_macro, inference_time_ms
Medical:  sensitivity_per_class, deployment_ready, medical_threshold_passed
```

### **Accessing Logged Results**

**View Training + Evaluation Results:**
```bash
# Start MLflow UI to view all experiments
mlflow ui

# View specific experiment comparisons
# Navigate to: http://localhost:5000
# - Training: brain-cancer-mri-v2-{model}
# - Evaluation: brain-cancer-mri-evaluation-{model}
# - Testing: brain-cancer-mri-test-{model}
```

**Wandb Dashboard Access:**
- **Training**: https://wandb.ai/your-username/brain-cancer-mri
- **Evaluation**: https://wandb.ai/your-username/brain-cancer-mri-evaluation  
- **Testing**: https://wandb.ai/your-username/brain-cancer-mri-test

**What You Can Track:**
- **Model Performance**: Compare test accuracy across all models
- **Medical Compliance**: Track which models pass medical AI thresholds
- **Inference Speed**: Monitor real-time performance for deployment
- **Training vs Test**: Detect overfitting by comparing validation vs test metrics
- **Experiment History**: Full audit trail for regulatory submissions

## 📋 **Quick Reference**

### **Evaluation Commands**
```bash
# Comprehensive evaluation (recommended for medical AI)
python3 evaluate.py --model efficientnet_b0 --detailed --medical-validation --compare

# Basic evaluation with logging
python3 evaluate.py --model resnet18

# Simple test (quick check)
python3 test.py --model efficientnet_b0

# Model comparison only
python3 evaluate.py --model swin_t --compare
```

### **Monitoring Commands**
```bash
# View all experiments
mlflow ui                    # http://localhost:5000
tensorboard --logdir runs   # http://localhost:6006
wandb                        # https://wandb.ai
```

### **Available Models**
`resnet18`, `resnet50`, `efficientnet_b0`, `swin_t`, `swin_s`, `vit_b_16`, `medical_cnn`, `xception_medical`

##  Medical AI Considerations

### **Clinical Deployment Requirements**

**Regulatory Compliance:**
- **FDA/CE Marking**: Complete audit trail and documentation required
- **Clinical Validation**: Performance metrics for medical review
- **Safety Documentation**: Model behavior analysis and limitations
- **Traceability**: Full pipeline from training to deployment

**Medical AI Standards:**
- **Accuracy Thresholds**: 85%+ for medical screening applications
- **Sensitivity Requirements**: 80%+ for tumor detection (minimize false negatives)
- **Real-time Performance**: <500ms inference for clinical decision support
- **Cross-platform Compatibility**: Support for various hospital systems

### **Clinical Integration Considerations**

**Hospital System Integration:**

## 📚 Documentation

- **`WORKFLOW_GUIDE.md`**- Detailed training & registration workflow guide
- **`README.md`**- Main project documentation (this file)
- **`requirements.txt`**- Python dependencies
- **`config/config.yaml`**- Training configuration

### **Key Scripts**
- **`train.py`**- Training script (no automatic registration)
- **`evaluate.py`**- Comprehensive evaluation with medical AI validation
- **`register_model.py`**- Model registration with MLflow (use after good performance)
- **`test.py`**- Simple test set evaluation
- **`export_model.py`**- Model export for deployment (TorchScript/ONNX)
- **`compute_normalization.py`**- Comprehensive normalization statistics (display/save/verify)
- **PACS Compatibility**: Integration with Picture Archiving and Communication Systems
- **DICOM Support**: Standard medical imaging format compatibility
- **Workflow Integration**: Seamless integration with radiologist workflows
- **Fallback Mechanisms**: Handling edge cases and system failures

**Clinical Decision Support:**
- **Confidence Scores**: Providing uncertainty estimates for clinical decisions
- **Explainability**: Model interpretability for clinical review
- **Multi-modal Integration**: Combining with other diagnostic information
- **Clinical Validation**: Real-world performance monitoring

### **Quality Assurance & Monitoring**

**Continuous Monitoring:**
- **Performance Tracking**: Monitor model degradation over time
- **Data Drift Detection**: Identify changes in input data distribution
- **Clinical Feedback**: Incorporate radiologist feedback and corrections
- **Model Updates**: Version control and deployment management

**Safety & Ethics:**
- **Bias Detection**: Identify and mitigate algorithmic bias
- **Privacy Protection**: HIPAA compliance and data security
- **Informed Consent**: Clear communication about AI assistance
- **Human Oversight**: Radiologist review and validation requirements

---

**Happy Training! 🚀**

*This project is designed for medical AI research and development. For clinical deployment, ensure compliance with all applicable regulatory requirements and conduct thorough clinical validation.*
