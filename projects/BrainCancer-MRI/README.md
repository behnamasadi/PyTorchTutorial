# Brain Cancer MRI Classification

A PyTorch-based deep learning project for classifying brain MRI images to detect different types of brain tumors using ResNet architecture.

## 🎯 Project Overview

This project implements a convolutional neural network to classify brain MRI images into three categories:
- **Glioma Tumor**
- **Meningioma Tumor** 
- **Pituitary Tumor**

## 🏗️ Architecture

- **Models**: Multiple architectures supported (ResNet, Swin Transformer, EfficientNet, Vision Transformer)
- **Framework**: PyTorch 2.0+ with compilation support
- **Monitoring**: TensorBoard + MLflow + Weights & Biases + Hardware monitoring
- **Performance**: Mixed precision training, GPU optimization, multi-core data loading
- **Evaluation**: Comprehensive medical AI validation with clinical deployment assessment
- **Data Augmentation**: Random horizontal flip, rotation, color jitter

## 📁 Project Structure

```
BrainCancer-MRI/
├── config/
│   └── config.yaml          # Training configuration
├── data/
│   └── dataset.py           # Dataset loading and preprocessing
├── models/
│   └── model.py             # Model architecture
├── utils/
│   ├── helpers.py           # Utility functions
│   └── path_utils.py        # Path management
├── train.py                 # Main training script
├── evaluate.py              # Comprehensive model evaluation
├── test.py                  # Simple test set evaluation
├── export_model.py          # Model export for deployment
├── run_experiments.py       # Automated multi-model comparison
├── checkpoints/             # Model checkpoints (created during training)
├── runs/                    # TensorBoard logs (created during training)
├── mlruns/                  # MLflow logs (created during training)
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Required packages
pip install torch torchvision
pip install tensorboard mlflow wandb
pip install pyyaml pillow psutil GPUtil onnx
```

### Training

**Simple training with default settings:**
```bash
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

**Run multiple experiments:**
```bash
# Compare different models automatically
python run_experiments.py
```

### Evaluation

**Comprehensive model evaluation with full logging:**
```bash
# Basic evaluation (with MLflow + Wandb logging)
python3 evaluate.py --model efficientnet_b0

# Full medical AI analysis (comprehensive logging)
python3 evaluate.py --model efficientnet_b0 --detailed --medical-validation --compare

# Quick test (simple logging)
python3 test.py --model efficientnet_b0
```

**📊 Evaluation Logging Output:**
```
📊 Setting up evaluation logging...
📊 MLflow experiment: brain-cancer-mri-evaluation-efficientnet_b0
🔮 Wandb project: brain-cancer-mri-evaluation
🏆 Model shows excellent performance for medical AI!
```

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

## 📊 Monitoring & Visualization

This project implements **comprehensive logging** across the entire ML pipeline: training, validation, and testing.

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
- **Hardware utilization metrics** (GPU/CPU/Memory usage)
- **Performance metrics** (batch processing time, throughput)

**🧪 Evaluation Logs:**
- **Test set performance**: Accuracy, precision, recall, F1-score
- **Per-class metrics**: Detailed analysis for each tumor type
- **Medical AI validation**: Clinical deployment readiness
- **Inference performance**: Real-time timing and throughput
- **Visualizations**: Confusion matrices and performance charts
- **Model comparisons**: Multi-model performance rankings

### 🔗 **End-to-End Traceability**

**Complete ML Pipeline Logging:**
1. **Training** → Log hyperparameters, validation metrics, checkpoints
2. **Evaluation** → Log test performance, medical validation, artifacts
3. **Deployment** → Link training runs to final test results

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

#### **⚡ Performance Optimization**
- [x] **Mixed Precision Training**: FP16 for 2x faster training and 50% memory savings
  - ⚠️ **Medical Image Warning**: Disable for medical images due to numerical instability
- [x] **Model Compilation**: PyTorch 2.0 compilation for 20-30% speed boost
  - ⚠️ **Stability Note**: Disable if encountering training issues
- [x] **GPU Memory Optimization**: Channels-last memory format, CUDNN benchmarking
- [x] **Multi-core Data Loading**: 8 workers with prefetching and persistent workers
- [x] **Gradient Clipping**: Prevents gradient explosions for stable training

#### **🏥 Medical Image Specific Settings**
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
- **Reduce batch size** for larger models (Swin-S, ViT)
- **Disable `channels_last`** if memory-constrained: `channels_last: false`
- **Disable model compilation** for debugging: `compile_model: false`
- **Use gradient checkpointing** for very large models
- **Monitor GPU usage**: `nvidia-smi` during training

#### **⏰ Early Stopping & Patience**

The training now includes **automatic early stopping** to prevent overfitting:

```yaml
train:
  epochs: 50
  patience: 10          # Stop if no improvement for 10 epochs
  min_delta: 0.001      # Minimum improvement threshold (0.1%)
```

**How it works:**
- **Tracks best validation accuracy** across all epochs
- **Saves best model** automatically when validation improves
- **Stops training early** if no improvement for `patience` epochs
- **Prevents overfitting** and saves training time
- **Logs improvement details** with each new best model

**Example output:**
```
💾 New best model saved! Val Acc: 87.45% (improvement: 0.234%)
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

#### **📊 Monitoring & Tracking**
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

#### **📊 Comprehensive Evaluation**
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
📊 Dataset: 4239 train, 908 validation samples
🏗️  Model: resnet18 (resnet18) with 3 classes
📦 Batch size: 128 (model-optimized)
📈 Learning rate: 0.001
🔄 Epochs: 50
🎯 Device: cuda
💻 CPU: 20 cores @ 8.5%
🧠 RAM: 31.7% used
🚀 GPU: NVIDIA GeForce RTX 3050
📊 GPU Usage: 95.2% | Memory: 78.5% (3140/4096MB)
⚡ Mixed Precision: Enabled
🔥 Model Compilation: Enabled
📈 TensorBoard: ./runs
📊 MLflow: brain-cancer-mri
🔮 Wandb: brain-cancer-mri
============================================================

📚 Epoch 1/50
--------------------------------------------------
  Batch 10/34 | Loss: 1.0483 | Acc: 46.95%
  Batch 20/34 | Loss: 0.9310 | Acc: 57.19%
  Batch 30/34 | Loss: 0.8457 | Acc: 63.28%

📊 Epoch 1 Summary:
  🕐 Time: 75.78s
  📉 Train Loss: 0.8141 | Train Acc: 65.13%
  📊 Val Loss: 0.7442 | Val Acc: 63.88%
💾 Saving checkpoint at epoch 1
```

## 🤖 Available Models

| Model | Type | Parameters | Input Size | Batch Size* | Optimizer | Memory | Learning Rate |
|-------|------|------------|------------|-------------|-----------|--------|---------------|
| `resnet18` | CNN | 11.7M | 224×224 | **128** | AdamW | ~2.5GB | 0.0001 |
| `resnet50` | CNN | 25.6M | 224×224 | **64** | AdamW | ~3.5GB | 0.001 |
| `efficientnet_b0` | CNN | 5.3M | 224×224 | **128** | AdamW | ~2.0GB | 0.001 |
| `swin_t` | Transformer | 28.3M | 224×224 | **32** | AdamW | ~3.0GB | 0.00005 |
| `swin_s` | Transformer | 49.6M | 224×224 | **8** | AdamW | ~3.8GB | 0.0001 |
| `vit_b_16` | Transformer | 86.6M | 224×224 | **16** | AdamW | ~3.8GB | 0.0001 |
| `medical_cnn` | Medical CNN | 5.3M (frozen) + 0.2M | 224×224 | **64** | AdamW | ~2.0GB | 0.001 |
| **`xception_medical`** | **Kaggle Solution** | **5.3M (trainable)** | **299×299** | **16** | **Adamax** | **~2.2GB** | **0.001** |

*Optimized for RTX 3050/4GB VRAM. Automatically scales with available GPU memory.

### Model Selection Guide

**🚀 For Quick Experiments:**
- `resnet18`: Fast training, largest batch size (128), good baseline
- `efficientnet_b0`: Best efficiency/accuracy trade-off, large batches

**🎯 For Best Accuracy:**
- **`xception_medical`**: **Kaggle-proven solution** (299×299, Adamax optimizer)
- `swin_t`: Modern transformer architecture, optimized batch size
- `swin_s`: Higher capacity for complex patterns

**🏥 For Medical Domain:**
- **`xception_medical`**: **Kaggle brain tumor solution** with 99% reported accuracy
- `medical_cnn`: EfficientNet + frozen backbone, fast convergence
- `resnet18`: Reliable baseline for medical images

**⚡ For Production:**
- `efficientnet_b0`: Lightweight, efficient, maximum throughput
- `resnet18`: Well-tested, reliable, fastest training
- `xception_medical`: Proven medical imaging performance

**🔥 For Maximum GPU Utilization:**
- Models automatically use optimized batch sizes for your hardware
- Early stopping prevents overfitting and saves training time
- Mixed precision disabled for medical image stability
- All models benefit from hardware acceleration

## 🎛️ Customization

### Model Configuration
- Choose from 6 different architectures
- Model-specific learning rates automatically applied
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

**Example Output:**
```
🧠 Brain Cancer MRI Classification - Model Comparison
============================================================
🔬 Experiment 1/4
🚀 Starting experiment: resnet18
...
📊 EXPERIMENT SUMMARY
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
```bash
# Install all dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tensorboard mlflow wandb psutil GPUtil pyyaml pillow numpy
```

## ⚡ Performance Optimization

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

### 📊 **Performance Monitoring**

Real-time tracking of:
- GPU utilization and memory usage
- CPU usage across all cores  
- System memory consumption
- GPU temperature monitoring
- Batch processing throughput

### 🎯 **Model-Specific Optimizations**

| Model | Optimized Batch | Expected Speed | Memory Usage |
|-------|----------------|----------------|--------------|
| ResNet18 | 128 | **Fastest** | 2.5GB |
| EfficientNet-B0 | 128 | **Most Efficient** | 2.0GB |
| ResNet50 | 64 | Fast | 3.5GB |
| Swin-T | 32 | Moderate | 3.0GB |
| ViT-B/16 | 16 | Slower | 3.8GB |

### 💡 **Performance Tips**

1. **For Maximum Speed**: Use `resnet18` or `efficientnet_b0`
2. **For Memory Efficiency**: Enable mixed precision (default)
3. **For Large Datasets**: Increase `num_workers` to match CPU cores
4. **For Multiple GPUs**: Models support DataParallel automatically

## 🐛 Troubleshooting

### Common Issues

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


## 🎯 **Pre-deployment Quality & Validation**

After successfully training your model, follow these essential steps before deploying to production:

### **1. 📊 Comprehensive Model Evaluation**

Use the new **comprehensive evaluation script** for detailed analysis:

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
- **Detailed evaluation report** (`evaluation_report.txt`)
- **JSON metrics** (`evaluation_metrics.json`) for programmatic access
- **Enhanced confusion matrix** with counts and percentages
- **Per-class metrics visualization** (precision, recall, F1)
- **Medical AI validation report** (`medical_validation.json`)
- **Multi-model comparison** with performance rankings

### **1b. 📊 Simple Test Set Evaluation**

For quick testing, use the simple evaluation script:

```bash
python3 test.py --model efficientnet_b0
python3 test.py --model resnet18  
python3 test.py --model xception_medical
```

This provides basic metrics and confusion matrix.

### **2. 🔍 Medical AI Validation Features**

The `evaluate.py` script includes specialized medical AI validation:

**🏥 Clinical Standards:**
- **85%+ accuracy threshold** for medical screening deployment
- **80%+ sensitivity** for tumor detection (prevents missed diagnoses)
- **Specificity analysis** for each class (reduces false positives)
- **Clinical deployment readiness** assessment

**📊 Advanced Analysis:**
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
- **Accuracy rankings** across all trained models
- **F1-score analysis** for balanced performance assessment
- **Inference speed comparison** (ms per sample)
- **Medical AI compliance** status for each model
- **Best model recommendation** with rationale

### **4. ⚡ Performance Profiling**

Built-in performance analysis:

- **Real-time inference**: Automatic timing for each model (<1ms typical)
- **Memory efficiency**: Model-specific batch size optimization
- **Throughput analysis**: Batch processing capabilities
- **Hardware utilization**: GPU/CPU usage monitoring during evaluation

### **5. 📊 Evaluation Output Examples**

The `evaluate.py` script generates comprehensive outputs:

**📋 Console Output:**
```
🧠 Brain Cancer MRI Model Evaluation
==================================================
📋 Model: efficientnet_b0 (efficientnet_b0)
📊 Checkpoint validation accuracy: 91.63%
🖥️  Using device: cuda

🚀 Starting evaluation...
🧪 Running inference on test set...
⏱️  Total evaluation time: 2.20s

✅ **EVALUATION RESULTS**
🎯 Test Accuracy: 0.9230 (92.30%)
📊 Macro F1-Score: 0.9238
⚡ Avg inference time: 0.82ms per sample

🏥 MEDICAL AI VALIDATION RESULTS:
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

**📊 Logged to Monitoring Platforms:**
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
🎯 Loaded model from epoch 6
📊 Validation accuracy: 0.9163
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

- [ ] **Test accuracy ≥ 85%** (medical AI threshold) ✅ *Auto-checked*
- [ ] **Per-class recall ≥ 80%** (no missed diagnoses) ✅ *Auto-checked*
- [ ] **Inference latency < 500ms** ✅ *Auto-measured*
- [ ] **Sensitivity warnings addressed** ✅ *Auto-reported*
- [ ] **Class imbalance < 5:1 ratio** ✅ *Auto-detected*
- [ ] **Confusion matrix analysis** ✅ *Auto-generated*
- [ ] **Model comparison completed** ✅ *Auto-compared*
- [ ] **Medical validation passed** ✅ *Auto-validated*

**Quick validation:**
```bash
python3 evaluate.py --model your_model --medical-validation
```

### **🏥 Medical AI Specific Considerations**

- **FDA/Regulatory compliance**: Document training data, validation methodology
- **Clinical integration**: Ensure compatibility with PACS systems
- **Radiologist workflow**: Design UI for clinical decision support
- **Audit trail**: Log all predictions for regulatory review
- **Fallback mechanisms**: Handle edge cases gracefully

### **📊 Evaluation Logging Best Practices**

**Why Log Evaluation Results?**

**🏥 Medical AI Requirements:**
- **Regulatory compliance**: FDA/CE marking requires complete audit trails
- **Clinical validation**: Document test performance for medical review
- **Model versioning**: Track which model version achieved what performance
- **Safety documentation**: Prove model safety and efficacy

**📈 MLOps Benefits:**
- **Model registry**: Link training experiments to final test performance
- **Performance monitoring**: Track model degradation over time
- **A/B testing**: Compare model performance across different test sets
- **Deployment decisions**: Use logged metrics to select production models

**🔄 Complete Pipeline Traceability:**
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

### **🔍 Accessing Logged Results**

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

**📊 What You Can Track:**
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

---

**Happy Training! 🚀**
