# Brain Cancer MRI Classification

A comprehensive deep learning project for "Lungs Disease Dataset" classification using chest X-Rays images, featuring state-of-the-art architectures, medical AI validation.

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Training & Monitoring](#training--monitoring)


## Project Overview

This project implements a various convolutional neural network to classify chest X-Rays into the following categories:

- **Viral Pneumonia**
- **Bacterial Pneumonia**
- **Covid**
- **Tuberculosis**
- **Normal**

[Dataset on Kaggle](https://www.kaggle.com/datasets/khaleddev/lungs-disease-dataset-broken)

#### Automatic Dataset Download

The training script automatically downloads the dataset from Kaggle if it doesn't already exist. No manual download required!

**How it works:**
- Checks if `data/train` and `data/val` directories already exist
- If missing, automatically downloads from Kaggle using `kagglehub`
- Organizes the data into the correct train/val/test structure
- Uses cached downloads if available (no re-download needed)

**Authentication:**

The script supports multiple authentication methods:

1. **Environment Variables (RunPod/Docker):**
   ```bash
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   ```

2. **Local Development:**
   - Create `~/.kaggle/kaggle.json` with your credentials:
     ```json
     {
       "username": "your_username",
       "key": "your_api_key"
     }
     ```
   - Or use `kagglehub.login()` in Python

The script automatically detects which method to use and provides clear status messages.



### **Key Features**
- **Multiple Architectures**: ConvNeXt, EfficientNetV2, RegNetY
- **Medical AI Optimized**: Model-specific configurations for clinical deployment
- **Transfer Learning**: Pre-trained weights for robust medical image classification
- **Production Ready**: Complete MLflow model registry and deployment pipeline
- **Comprehensive Monitoring**: TensorBoard + MLflow + Weights & Biases integration

## Architecture


#### **Model Architecture & Training Strategy**


| Model | Architecture | Pre-trained Backbone | Trainable Parts | Training Strategy | Optimizer |
|-------|--------------|---------------------|-----------------|-------------------|-----------|
| **ConvNeXtV2-Tiny** | CNN | ImageNet weights | Final classifier only | Transfer Learning | AdamW |
| **ConvNeXtV2-Base** | CNN | ImageNet weights | Final classifier only | Transfer Learning | AdamW |
| **EfficientNetV2-L** | CNN | ImageNet weights | Final classifier only | Transfer Learning | AdamW |
| **EfficientNetV2-M** | CNN | ImageNet weights | Final classifier only | Transfer Learning | AdamW |
| **EfficientNetV2-S** | CNN | ImageNet weights | Final classifier only | Transfer Learning | AdamW |
| **RegNetY-4GF** | CNN | ImageNet weights | Final classifier only | Transfer Learning | AdamW |
| **RegNetY-8GF** | CNN | ImageNet weights | Final classifier only | Transfer Learning | AdamW |

### **Medical Fine-Tuning Strategy - Detailed Training Logic**

Since medical images (chest X-rays) are fundamentally different from ImageNet natural images, this project uses a **two-stage medical fine-tuning policy** - the most standard approach in medical imaging.

#### **Training Flow Overview**

The training process follows this sequence:

```
1. Load pre-trained model (ImageNet weights)
2. Replace classifier head with custom head for 5 classes
3. Stage 1: Freeze backbone, train only head
4. Stage 2: Unfreeze backbone, train entire model with differential learning rates
5. Save best checkpoint based on validation accuracy
```

#### **Stage 1 - Freeze Backbone, Train Classifier Only**

**Configuration (from `train_runpod.yaml`):**
```yaml
stage1:
  enabled: true
  freeze_backbone: true
  epochs: 5
  learning_rate: 1e-4
```

**What Happens:**
1. **Parameter Freezing**: All backbone parameters are frozen (`requires_grad=False`)
   - Only the classifier head remains trainable
   - Typically ~1-5% of total parameters are trainable
   
2. **Optimizer**: Uses **Adam** optimizer
   - Simple optimizer sufficient for small number of parameters
   - Single learning rate: `1e-4` (0.0001)
   - No learning rate scheduling (constant LR)

3. **Training Process**:
   - Runs for 5 epochs (configurable)
   - Only head parameters are updated
   - Backbone features remain frozen (preserves ImageNet knowledge)

**Purpose:** 
- Initialize the classifier with task-specific features
- Prevent early overfitting to medical domain specifics
- Stabilize training before fine-tuning the backbone
- Quick warm-up phase (typically 5-10 epochs)

**Example Output:**
```
➡️  Stage 1: Freezing backbone
  Trainable parameters: 165,120 / 28,000,000 (0.59%)
  Optimizer: Adam with lr=1.00e-04
Stage 1 | Epoch 1/5 | Train Loss 1.2345 Acc 45.23% | Val Loss 0.9876 Acc 52.34% | LR 1.00e-04
...
```

#### **Stage 2 - Unfreeze Backbone, End-to-End Training with Differential Learning Rates**

**Configuration (from `train_runpod.yaml`):**
```yaml
stage2:
  enabled: true
  freeze_backbone: false
  epochs: 15
  learning_rate: 3e-5  # Base LR (used for head if head_lr not specified)
  head_lr: 3e-5        # Head learning rate
  backbone_lr: 5e-6    # Backbone LR (6x smaller to preserve pretrained features)
  early_stop_patience: 5
  lr_schedule: "cosine"
  lr_schedule_params:
    T_max: 15
    eta_min: 1e-7
```

**What Happens:**
1. **Parameter Unfreezing**: All parameters become trainable
   - Backbone: ~95-99% of parameters
   - Head: ~1-5% of parameters
   - All parameters now participate in gradient updates

2. **Optimizer**: Uses **AdamW** optimizer with **differential learning rates**
   - **Backbone LR**: `5e-6` (0.000005) - Very small to preserve ImageNet features
   - **Head LR**: `3e-5` (0.00003) - 6x larger than backbone
   - **Weight Decay**: 0.01 (regularization)
   
   **Why Differential Learning Rates?**
   - Pre-trained backbone features are valuable and should change slowly
   - Head needs to adapt faster to the new task
   - Prevents "catastrophic forgetting" of ImageNet knowledge
   - Standard practice in transfer learning

3. **Learning Rate Scheduling**: **Cosine Annealing**
   - Starts at configured LR (3e-5 for head, 5e-6 for backbone)
   - Gradually decreases following cosine curve
   - Reaches minimum LR (`eta_min=1e-7`) at the end
   - Formula: `lr(t) = eta_min + (lr_initial - eta_min) * (1 + cos(π * t / T_max)) / 2`
   - `T_max=15` (number of epochs)

4. **Early Stopping**: Monitors validation loss
   - Stops if no improvement for 5 consecutive epochs
   - Prevents overfitting
   - Only active in Stage 2 (not Stage 1)

5. **Checkpoint Saving**: 
   - Saves checkpoint whenever validation accuracy improves
   - Checkpoint format: `{model_name}-training_best_model.pth`
   - Contains: `state_dict`, `epoch`, `val_accuracy`, `stage`

**Purpose:**
- Adapt pre-trained ImageNet features to medical domain
- Fine-tune low-level features (edges, textures) for medical patterns
- Achieve better domain-specific feature representation
- Balance between adaptation and preservation of pre-trained knowledge

**Example Output:**
```
➡️  Stage 2: Unfreezing backbone
  Trainable parameters: 28,000,000 / 28,000,000 (100.00%)
  Optimizer: AdamW with separate LRs
    Backbone LR: 5.00e-06 (27,835,000 params)
    Head LR: 3.00e-05 (165,120 params)
Stage 2 | Epoch 1/15 | Train Loss 0.8234 Acc 72.45% | Val Loss 0.7123 Acc 78.90% | LR 3.00e-05
💾 Saved new best checkpoint (Val Acc 78.90%)
...
⏹️  Early stopping triggered: No improvement in val loss for 5 epochs
   Best val loss: 0.6543, Best val acc: 84.47%
```

#### **Complete Training Configuration**

**Default Settings (from `train_runpod.yaml`):**

| Setting | Stage 1 | Stage 2 |
|---------|---------|---------|
| **Epochs** | 5 | 15 |
| **Backbone** | Frozen | Unfrozen |
| **Optimizer** | Adam | AdamW |
| **Learning Rate** | 1e-4 (single) | Head: 3e-5, Backbone: 5e-6 |
| **LR Schedule** | None (constant) | Cosine Annealing |
| **Weight Decay** | 0 | 0.01 |
| **Early Stopping** | Disabled | Enabled (patience=5) |
| **Loss Function** | CrossEntropyLoss with label_smoothing=0.1 | Same |

**Total Training Time**: ~20 epochs (5 + 15), but may stop earlier due to early stopping

#### **Why Two-Stage Training?**

Medical images (X-rays, CT scans, MRIs) have fundamentally different characteristics than natural images:
- **Different textures**: Medical images have distinct patterns (bone structures, soft tissues, anomalies)
- **Different semantics**: Features learned on ImageNet (cats, cars, objects) need adaptation
- **Different scales**: Medical anomalies can be subtle and require fine-grained feature learning
- **Domain gap**: Large gap between natural and medical images requires gradual adaptation

**Benefits of Two-Stage Approach:**
1. **Stability**: Stage 1 stabilizes the classifier before fine-tuning
2. **Preservation**: Differential LRs in Stage 2 preserve valuable ImageNet features
3. **Efficiency**: Faster convergence than training from scratch
4. **Robustness**: Less prone to overfitting on small medical datasets

This two-stage approach is the **universal "medical fine-tuning" recipe** used in:
- Medical imaging research papers
- Clinical deployment pipelines
- FDA-approved medical AI systems

#### **Learning Rate Strategy Details**

**Stage 1 - Single Learning Rate:**
- All trainable parameters (head only) use the same LR: `1e-4`
- Simple and effective for small parameter set
- No scheduling needed (short training phase)

**Stage 2 - Differential Learning Rates:**
- **Backbone LR** (`5e-6`): 6x smaller than head LR
  - Protects pre-trained ImageNet features
  - Allows gradual adaptation to medical domain
  - Prevents catastrophic forgetting
  
- **Head LR** (`3e-5`): 6x larger than backbone LR
  - Allows faster adaptation to new task
  - Head is randomly initialized, needs more learning
  - Can change more aggressively

**Ratio**: Head LR / Backbone LR = 6:1 (configurable)

**Cosine Annealing Schedule:**
- Starts at initial LR values
- Smoothly decreases following cosine curve
- Reaches `eta_min=1e-7` at epoch 15
- Provides smooth convergence without sudden drops

#### **Checkpoint Management**

**Checkpoint Naming:**
- Best model: `{model_name}-training_best_model.pth`
- Last model: `{model_name}-training_last_model.pth`
- Example: `tf_efficientnetv2_s-training_best_model.pth`

**Checkpoint Contents:**
```python
{
    'stage': 'Stage 2',
    'epoch': 12,
    'val_accuracy': 0.8447,
    'state_dict': {...}  # Model weights
}
```

**Saving Strategy:**
- Best checkpoint: Saved whenever validation accuracy improves
- Last checkpoint: Saved at the end of training (final epoch)
- Both uploaded to wandb as artifacts with model-specific names

#### **Loss Function & Regularization**

**CrossEntropyLoss with Label Smoothing:**
```yaml
loss:
  name: "CrossEntropyLoss"
  label_smoothing: 0.1
```

**Label Smoothing (0.1):**
- Prevents overconfident predictions
- Improves generalization on medical datasets
- Reduces overfitting to training data
- Standard practice for medical AI (especially with limited data)

**How it works:**
- Instead of hard labels [0, 0, 1, 0, 0], uses soft labels [0.025, 0.025, 0.9, 0.025, 0.025]
- Smoothing factor: 0.1 (10% of probability mass redistributed)
- Formula: `soft_label = (1 - smoothing) * hard_label + smoothing / num_classes`

#### **Training Loop Details**

**Per Epoch:**
1. **Training Phase:**
   - Model set to `train()` mode
   - Iterate through training batches
   - Forward pass → Compute loss → Backward pass → Optimizer step
   - Track training loss and accuracy

2. **Validation Phase:**
   - Model set to `eval()` mode
   - Iterate through validation batches (no gradients)
   - Compute validation loss and accuracy
   - Used for:
     - Early stopping decisions
     - Best checkpoint selection
     - Learning rate scheduling (if using ReduceLROnPlateau)

3. **Learning Rate Update:**
   - **Cosine Annealing**: Updated every epoch based on cosine schedule
   - **ReduceLROnPlateau**: Updated only when validation loss plateaus
   - Current LR logged to TensorBoard, wandb, and MLflow

4. **Checkpoint Management:**
   - If validation accuracy improved → Save best checkpoint
   - Checkpoint includes: model state, epoch, validation accuracy, stage name
   - Upload to wandb as artifact (with model-specific naming)

5. **Early Stopping Check (Stage 2 only):**
   - Monitor validation loss
   - If no improvement for `patience` epochs → Stop training
   - Prevents overfitting and saves compute time

6. **Logging:**
   - Metrics logged to TensorBoard, wandb, and MLflow
   - Separate metrics for Stage 1 and Stage 2
   - Includes: train_loss, train_acc, val_loss, val_acc, learning_rate

#### **Training Configuration Summary**

**Complete Training Pipeline:**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Load Pre-trained Model (ImageNet weights)                │
│    - Replace classifier head for 5 classes                   │
│    - Custom head: AdaptiveAvgPool2d → LayerNorm →           │
│                   Dropout(0.3) → Linear(128) →              │
│                   ReLU → Dropout(0.25) → Linear(5)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Stage 1: Freeze Backbone (5 epochs)                      │
│    - Freeze: All backbone parameters                        │
│    - Train: Only classifier head                            │
│    - Optimizer: Adam (lr=1e-4)                             │
│    - LR Schedule: None (constant)                          │
│    - Early Stopping: Disabled                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Stage 2: Unfreeze Backbone (15 epochs, early stop)       │
│    - Freeze: None (all parameters trainable)                │
│    - Optimizer: AdamW with differential LRs                 │
│      • Backbone LR: 5e-6 (preserve ImageNet features)       │
│      • Head LR: 3e-5 (faster adaptation)                   │
│    - LR Schedule: Cosine Annealing                          │
│      • T_max: 15 epochs                                     │
│      • eta_min: 1e-7                                        │
│    - Early Stopping: Enabled (patience=5)                   │
│    - Weight Decay: 0.01 (regularization)                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Save Checkpoints                                          │
│    - Best: {model}-training_best_model.pth                  │
│    - Last: {model}-training_last_model.pth                  │
│    - Upload to wandb with model-specific names              │
└─────────────────────────────────────────────────────────────┘
```

#### **Key Training Parameters**

| Parameter | Stage 1 | Stage 2 | Purpose |
|-----------|---------|---------|---------|
| **Epochs** | 5 | 15 | Training duration |
| **Trainable Params** | ~0.6% | 100% | Which parameters update |
| **Optimizer** | Adam | AdamW | Optimization algorithm |
| **Backbone LR** | N/A (frozen) | 5e-6 | Preserve ImageNet features |
| **Head LR** | 1e-4 | 3e-5 | Adapt to new task |
| **LR Ratio** | N/A | 6:1 (head:backbone) | Differential learning |
| **LR Schedule** | None | Cosine | Smooth convergence |
| **Weight Decay** | 0 | 0.01 | Regularization |
| **Early Stopping** | No | Yes (patience=5) | Prevent overfitting |
| **Label Smoothing** | 0.1 | 0.1 | Improve generalization |

#### **Why This Configuration Works**

1. **Stage 1 (Freeze + Low LR)**: 
   - Stabilizes classifier without disrupting backbone
   - Quick warm-up (5 epochs)
   - Prevents early overfitting

2. **Stage 2 (Differential LRs)**:
   - Backbone changes slowly (5e-6) → Preserves ImageNet knowledge
   - Head changes faster (3e-5) → Adapts to medical task
   - 6:1 ratio is optimal for medical transfer learning

3. **Cosine Annealing**:
   - Smooth LR decay prevents sudden drops
   - Helps model converge to better local minima
   - Standard for fine-tuning scenarios

4. **Early Stopping**:
   - Prevents overfitting on small medical datasets
   - Saves compute time
   - Model typically peaks around epoch 7-13

5. **Label Smoothing**:
   - Critical for medical datasets (often limited data)
   - Reduces overconfidence
   - Improves calibration of predictions

### **Grayscale Image Handling**

Most medical images (X-rays, CT scans) are grayscale, but pre-trained models expect RGB (3 channels). The standard approach:

**Repeat grayscale → 3 channels:**

```python
# Convert grayscale image to 3-channel RGB
# Simply repeat the single channel 3 times
grayscale_image = image  # Shape: [H, W] or [1, H, W]
rgb_image = grayscale_image.repeat(3, axis=0)  # Shape: [3, H, W]

# Or using transforms:
transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Repeat to 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[...], std=[...])
])
```

**Why this approach?**

- Pre-trained models were trained on 3-channel RGB images
- Repeating grayscale channels preserves the single-channel information
- Allows direct use of ImageNet pre-trained weights without architecture changes
- Standard practice in medical imaging pipelines

**Alternative approaches** (less common):
- Train grayscale models from scratch (requires more data)
- Use medical-specific pre-trained models (limited availability)
- Custom 1-channel input layers (requires architecture modification)

The "repeat grayscale to 3 channels" approach is the **most common and recommended** method in medical imaging.

### Normalization vs. Input Resolution

- Training images arrive at wildly different native resolutions (e.g., 2297×2032, 1790×1140, 512×512). The normalization script (`python -m lung_disease_dataset.utils.compute_normalization`) first resizes every sample using the same preprocessing pipeline that the models use (`dataset.image_size` in `configs/data.yaml`) before it measures statistics.  
- Because of that resize step, the reported mean/std values in `configs/data.yaml` always correspond to the target training resolution, not to the original pixel dimensions.  
- If you change a model’s expected input size (for example, ConvNeXt at 224 vs. EfficientNet at 384), update `dataset.image_size` and re-run the normalization script so the stored mean/std matches the new preprocessing.  
- All models that share a particular `image_size` can safely reuse the same normalization constants, even if the raw datasets contained mixed resolutions.

## Project Structure

This project follows professional ML project structure best practices used by Google Brain, Meta FAIR, HuggingFace, PyTorch Lightning, NVIDIA NeMo, and OpenAI academic projects.

### Directory Layout

```
Lung_Disease_Dataset/
├── pyproject.toml          # Project metadata and dependencies
├── .gitignore              # Git ignore rules
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies (legacy, use pyproject.toml)
│
├── src/
│   └── lung_disease_dataset/   # Main Python package
│       ├── __init__.py
│       ├── core/                # Core training/evaluation logic
│       │   ├── __init__.py
│       │   ├── training.py
│       │   └── evaluation.py
│       ├── data/                # Data loading utilities
│       │   └── __init__.py
│       ├── models/              # Model definitions
│       │   ├── __init__.py
│       │   └── model.py
│       └── utils/               # Utility functions
│           ├── __init__.py
│           ├── paths.py         # Stable path utilities
│           ├── compute_normalization.py
│           ├── normalization_constants.py
│           └── files_utility/
│
├── scripts/                # Executable scripts
│   ├── train.py
│   ├── evaluate.py
│   └── test.py
│
├── configs/                # Configuration files
│   └── config.yaml
│
├── notebooks/              # Jupyter notebooks
│   └── kaggle.ipynb
│
├── data/                   # Dataset files (gitignored)
│   ├── train/
│   ├── val/
│   └── test/
│
├── checkpoints/            # Model checkpoints (gitignored)
├── logs/                   # Log files (gitignored)
├── runs/                   # TensorBoard logs (gitignored)
├── mlruns/                 # MLflow logs (gitignored)
├── mlartifacts/            # MLflow artifacts (gitignored)
├── wandb/                  # Weights & Biases logs (gitignored)
├── outputs/                # Output files (gitignored)
├── results/                # Evaluation results (gitignored)
│
└── .vscode/                # VS Code settings
    ├── settings.json
    └── launch.json
```

### Key Directories

#### `src/lung_disease_dataset/` — Main Python Package

Contains all reusable code that becomes importable when you run `pip install -e .`:

- **`core/`** — Core training and evaluation logic
  - `training.py` — Training loop, optimizer setup, checkpoint saving
  - `evaluation.py` — Model evaluation, metrics calculation, visualization
  
- **`models/`** — Model architecture definitions
  - `model.py` — Model builder using timm library
  
- **`data/`** — Data loading utilities
  - Dataset loaders, transforms, preprocessing
  
- **`utils/`** — Utility functions
  - `paths.py` — Stable path utilities that work regardless of working directory
  - Other helper functions

#### `scripts/` — Executable Scripts

Contains thin orchestrator scripts that import and call functions from `src/lung_disease_dataset/`:

- `train.py` — Training script entry point
- `evaluate.py` — Model evaluation script
- `test.py` — Simple test script

#### `configs/` — Configuration Files

All YAML configuration files for hyperparameters, model settings, and experiment metadata.

### Import Examples

#### In Scripts (`scripts/train.py`, etc.)

```python
from lung_disease_dataset.models.model import get_model
from lung_disease_dataset.utils import project_root, resource_path
from lung_disease_dataset.core import train

# Load config using stable path utility
config_path = resource_path("configs", "config.yaml")
```

#### In Notebooks

```python
from lung_disease_dataset.models.model import get_model
from lung_disease_dataset.utils import resource_path

# Load data or configs
data_dir = resource_path("data")
```

#### In Package Code (`src/lung_disease_dataset/**`)

```python
from ..models.model import get_model
from ..utils import project_root
```

### Stable Path Utilities

The project uses `resource_path()` for all file paths, which works regardless of:
- Where VS Code launches the Python file
- Where Jupyter kernel was started
- Current working directory
- How Hydra changes your cwd
- Whether running from CLI, Docker, or pip-installed package

**Usage:**

```python
from lung_disease_dataset.utils import resource_path, project_root

# Get project root
root = project_root()

# Get paths relative to project root
data_path = resource_path("data", "train")
config_path = resource_path("configs", "config.yaml")
checkpoint_path = resource_path("checkpoints", "model.pt")
```

### VS Code Setup

The project includes `.vscode/settings.json` and `.vscode/launch.json` that:

- Set `PYTHONPATH` to include `./src` — enables imports to work correctly
- Configure debugging for scripts — press F5 to debug `train.py`, `evaluate.py`, etc.
- Enable auto-formatting with Black
- Configure linting with flake8

**Debug Configurations Available:**
- "Train" — Debug `scripts/train.py`
- "Evaluate" — Debug `scripts/evaluate.py`
- "Test" — Debug `scripts/test.py`
- "Python: Current File" — Debug any Python file

### Installation

Install the package in development mode:

```bash
cd /path/to/Lung_Disease_Dataset
pip install -e .
```

This makes `lung_disease_dataset` importable from anywhere, and any changes to source code are immediately reflected (no reinstall needed).

### Benefits of This Structure

1. **Reproducibility** — Stable paths work from anywhere (VS Code, Jupyter, CLI, Docker)
2. **Scalability** — Clear separation of concerns (core logic, data, models, utils)
3. **Professional** — Industry-standard structure used by top ML organizations
4. **VS Code Ready** — Fully configured for seamless development
5. **Packaging Ready** — Can be pip-installed, dockerized, or deployed
6. **Collaboration Friendly** — Easy for others to understand and contribute

## Training & Monitoring

### Quick Start - Running Scripts

All scripts can be run with default parameters. Navigate to the `scripts/` directory first:

```bash
cd scripts
```

#### Training

Train a model with default settings (uses `train_runpod.yaml` config):

```bash
python train.py
```

Train a specific model:

```bash
python train.py --config ../configs/train_runpod.yaml
```

Train with custom device:

```bash
python train.py --device cuda:0
```

#### Testing

**Process all available trained models** (automatically detects and processes all trained checkpoints):

```bash
python test.py
```

Test a specific model:

```bash
python test.py --model convnextv2_base
```

Test with custom config:

```bash
python test.py --config ../configs/eval.yaml --model tf_efficientnetv2_m
```

#### Comprehensive Evaluation

**Process all available trained models** (automatically detects and processes all trained checkpoints):

```bash
python evaluate.py
```

This will:
- Scan the checkpoints directory for all trained models
- Process each model sequentially
- Generate a summary comparison at the end showing which model performed best

Evaluate a specific model:

```bash
python evaluate.py --model convnextv2_base
```

Generate detailed analysis with visualizations:

```bash
python evaluate.py --model tf_efficientnetv2_s --detailed
```

Compare multiple models (when processing all models, comparison is automatic):

```bash
python evaluate.py --model tf_efficientnetv2_s --compare
```

Run medical AI validation checks:

```bash
python evaluate.py --model tf_efficientnetv2_s --medical-validation
```

**Note:** All scripts automatically:
- **Process all available models** when `--model` is not specified (scans for `{model}-training_best_model.pth` files)
- Find checkpoints with model-specific naming (`{model}-training_best_model.pth`)
- Extract model names from checkpoint filenames for accurate wandb reporting
- Load model configurations from `configs/model.yaml`
- Use appropriate image sizes and batch sizes for each model
- Show a summary comparison when processing multiple models

### Running Training

#### Local Development

For local development with smaller GPUs (e.g., 3-8GB VRAM):

```bash
cd scripts
python train.py --config ../configs/train_local.yaml
```

**Features:**
- Uses model-specific batch sizes from `model.yaml` (small values: 4-16)
- Optimized for local GPUs
- Lower `num_workers` (4) for typical local machines
- Uses local credentials (`~/.kaggle/kaggle.json` or `wandb login`)

#### RunPod/Cloud Deployment

For cloud GPUs with more VRAM (e.g., RTX 5090 with 32GB):

```bash
cd scripts
python train.py --config ../configs/train_runpod.yaml
```

**Features:**
- Model-specific batch sizes optimized for large GPUs (48-256 depending on model)
- Higher `num_workers` (12) for cloud instances
- Uses environment variables for authentication (see below)

**Default Configuration:**

If no config is specified, the script uses `train_runpod.yaml` by default:

```bash
cd scripts
python train.py  # Uses train_runpod.yaml
```

#### Custom Configuration

Specify a custom config file:

```bash
cd scripts
python train.py --config ../configs/train_local.yaml --device cuda:0
```

### Environment-Specific Batch Sizes

The project uses different batch sizes for different environments and models:

#### Local Development (`train_local.yaml`)
- Uses batch sizes from `model.yaml` (small values for local GPUs)
- Example: EfficientNetV2-M uses batch_size: 8

#### RunPod/Cloud (`train_runpod.yaml`)
- Model-specific batch sizes optimized for large GPUs (RTX 5090, 32GB VRAM):
  - ConvNeXtV2-Tiny: 256
  - EfficientNetV2-S: 192
  - EfficientNetV2-M: 144
  - EfficientNetV2-L: 96
  - ViT-Large: 48
  - And more...

**Batch Size Resolution Priority:**
1. `model_config` section in train config (highest priority)
2. `data.batch_size` override
3. `model.yaml` defaults
4. Fallback: 32

You can easily adjust batch sizes in `train_runpod.yaml` based on your GPU's VRAM.

### Experiment Tracking & Logging

The training script supports three logging backends for experiment tracking and visualization. All logging is **optional** — if a service is unavailable, training continues with a warning.

#### TensorBoard

**Local visualization of training metrics, loss curves, and model graphs.**

Start TensorBoard server:

```bash
tensorboard --logdir=runs
```

Then open your browser at: `http://localhost:6006`

**Default log directory:** `./runs` (configurable in `configs/train.yaml`)

#### MLflow

**Experiment tracking, model registry, and reproducibility.**

Start MLflow tracking server:

```bash
mlflow ui
```

Then open your browser at: `http://localhost:5000`

**Default artifacts directory:** `./mlruns` (configurable in `configs/train.yaml`)

**Features:**
- Parameter logging
- Metrics tracking
- Model versioning
- Experiment comparison
- Model artifact storage

#### Weights & Biases (wandb)

**Cloud-based experiment tracking with rich visualizations.**

**Local Development:**
Login once (credentials are cached):

```bash
wandb login
```

**RunPod/Docker:**
Use environment variable (no login needed):

```bash
export WANDB_API_KEY=your_api_key
```

Training will automatically log to your W&B account if configured in the training config file.

**Features:**
- Real-time metrics
- System monitoring (GPU, CPU, memory)
- Model checkpoints
- Hyperparameter sweeps
- Team collaboration

### Monitoring GPU Usage

Monitor GPU utilization during training:

```bash
watch -n 1 nvidia-smi
```

### Configuration

Edit `configs/train.yaml` to configure logging:

```yaml
monitoring:
  # TensorBoard
  tensorboard_log_dir: "./runs"
  
  # MLflow
  mlflow_tracking_uri: "./mlruns"
  mlflow_experiment_name: "lungs-disease"
  
  # Weights & Biases
  wandb:
    project: "Lungs Disease Dataset (4 types + normal)"
    entity: "your-username"  # Optional
    tags: ["medical", "x-ray", "classification"]
    notes: "Two-stage medical fine-tuning"
```

**Note:** If any logging service fails to connect (e.g., MLflow server not running), training will continue with a warning message.

## RunPod/Docker Deployment

The project is fully configured to work in both local development and cloud deployment environments (RunPod, Docker, etc.).

### Docker Setup

Run the training in a Docker container with GPU support:

```bash
docker run -it --runtime=nvidia \
  --entrypoint bash \
  -e KAGGLE_USERNAME=$KAGGLE_USERNAME \
  -e KAGGLE_KEY=$KAGGLE_KEY \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e HOME=/workspace \
  -v $HOME:/workspace \
  ghcr.io/behnamasadi/kaggle-projects:latest
```

Then inside the container:

```bash
python scripts/train.py --config configs/train_runpod.yaml
```

### Environment Variables

The following environment variables are supported for RunPod/Docker:

| Variable | Purpose | Local Alternative |
|----------|---------|-------------------|
| `KAGGLE_USERNAME` | Kaggle API username | `~/.kaggle/kaggle.json` |
| `KAGGLE_KEY` | Kaggle API key | `~/.kaggle/kaggle.json` |
| `WANDB_API_KEY` | Weights & Biases API key | `wandb login` |
| `HOME` | Home directory for cache/config | Default `~` |

### Dual Environment Support

The code automatically detects the environment and adapts:

**Local Environment:**
- Uses `~/.kaggle/kaggle.json` for Kaggle authentication
- Uses cached `wandb login` credentials
- Uses default `HOME` directory
- Shows: `💻 Running in local environment`

**RunPod/Docker Environment:**
- Uses `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables
- Uses `WANDB_API_KEY` environment variable
- Uses `HOME=/workspace` for cache/config
- Shows: `🌐 Running in RunPod/Docker environment`

**Automatic Fallback:**
- If environment variables are missing, falls back to local methods
- Clear status messages indicate which authentication method is being used
- Training continues even if some credentials are missing (with warnings)

### Configuration Files

The project includes environment-specific configuration files:

- **`configs/train_local.yaml`**: Optimized for local development
  - Smaller batch sizes (uses `model.yaml` defaults)
  - Lower `num_workers` (4)
  - Suitable for GPUs with 3-8GB VRAM

- **`configs/train_runpod.yaml`**: Optimized for cloud GPUs
  - Larger batch sizes (model-specific, 48-256)
  - Higher `num_workers` (12)
  - Optimized for RTX 5090 (32GB VRAM) and similar GPUs

Both configs share the same training strategy and hyperparameters, only batch sizes and data loading settings differ.

### Script Defaults Summary

| Script | Default Config | Default Model | Can Run Without Args? |
|--------|---------------|---------------|----------------------|
| `train.py` | `configs/train_runpod.yaml` | From config file | ✅ Yes |
| `test.py` | `configs/eval.yaml` | All available models (if `--model` not specified) | ✅ Yes |
| `evaluate.py` | `configs/eval.yaml` | All available models (if `--model` not specified) | ✅ Yes |

**Examples:**
```bash
cd scripts

# All work with defaults
python train.py
# Process all available trained models
python test.py
python evaluate.py

# Or specify a specific model
python test.py --model convnextv2_base
python evaluate.py --model tf_efficientnetv2_m --detailed

# Or specify model/config
python test.py --model convnextv2_base
python evaluate.py --model tf_efficientnetv2_m --detailed
```

### Available Models

The following models are available for training and evaluation:

- `convnextv2_tiny` - ConvNeXtV2-Tiny (224x224 input)
- `convnextv2_base` - ConvNeXtV2-Base (224x224 input)
- `tf_efficientnetv2_s` - EfficientNetV2-Small (288x288 input)
- `tf_efficientnetv2_m` - EfficientNetV2-Medium (320x320 input)
- `tf_efficientnetv2_l` - EfficientNetV2-Large (384x384 input)
- `regnety_004` - RegNetY-4GF (224x224 input)
- `regnety_006` - RegNetY-8GF (224x224 input)

### Model Checkpoints

Trained models are saved with model-specific naming:
- Best model: `{model_name}-training_best_model.pth`
- Last model: `{model_name}-training_last_model.pth`

Example: `tf_efficientnetv2_s-training_best_model.pth`, `convnextv2_base-training_best_model.pth`

Checkpoints are stored in the `checkpoints/` directory. Both `test.py` and `evaluate.py` automatically find checkpoints using this naming convention.

**Processing Multiple Models:** When you run `test.py` or `evaluate.py` without the `--model` argument, the scripts will:
1. Automatically scan the checkpoints directory for all trained models
2. Process each model sequentially
3. For `evaluate.py`, display a summary comparison at the end showing which model performed best

This makes it easy to evaluate all your trained models at once and compare their performance.
