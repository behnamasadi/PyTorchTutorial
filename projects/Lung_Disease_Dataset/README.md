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

### Evaluation Log Interpretation

This section provides a comprehensive guide to interpreting your evaluation logs and understanding what the results mean, why sensitivity warnings appear, and what conclusions you can draw.

#### 1. What the Evaluation Log Shows

Your evaluation script performs the following for each model:

1. Load the **best checkpoint** based on validation accuracy
2. Resize images according to the model's expected input size
3. Run inference on all **1954 test images**
4. Compute:
   - Overall test accuracy
   - Per-class sensitivity (recall)
   - Detailed confusion matrix
   - Classification report
   - JSON metrics
   - Visual plots (confusion matrix)
5. Trigger a warning if **sensitivity < 0.80** for any class

#### 2. Core Observation from the Log

Across ALL models (EffNet, ConvNeXt, RegNet), the evaluation shows:

**A. Test accuracy is excellent**

Top results:

| Model            | Test Accuracy |
| ---------------- | ------------- |
| EfficientNetV2-M | 89.20%        |
| EfficientNetV2-L | 88.69%        |
| ConvNeXtV2-Base  | 86.75%        |
| ConvNeXtV2-Tiny  | 85.36%        |

These results are fully consistent with training and test logs.

**B. Same two classes always show low sensitivity**

In every model:
- **class_0 sensitivity = 0.61 → 0.77**
- **class_4 sensitivity = 0.49 → 0.78**

This is exactly what triggered the warnings:

```
⚠️ WARNING: Low sensitivity for class_0 …
⚠️ WARNING: Low sensitivity for class_4 …
```

These two classes consistently underperform across all architectures.

#### 3. Why the Warnings Appear (Real Reason)

Your evaluation script flags sensitivity < 0.80 because:

$$\text{sensitivity} = \frac{TP}{TP + FN}$$

So low sensitivity = many **false negatives**, which is dangerous for clinical use.

**Why classes 0 and 4?**

Because (as you already noticed in training):
- They are the two pneumonia subclasses
- Pneumonia types **look extremely similar** on chest X-rays
- The dataset has **class imbalance** (viral pneumonia often underrepresented)
- Models confuse:
  - Viral ↔ Bacterial pneumonia
  - Viral ↔ Normal
  - Bacterial ↔ TB

This is NOT a model bug.

This is a **medical and dataset challenge**.

#### 4. Why Evaluation Accuracy Is Lower Than Validation Accuracy

Example:
- **ConvNeXtV2-Base**
  - Validation accuracy: 89.27%
  - Evaluation accuracy: 86.75%

This is completely normal.

**Why?**

**Validation data:**
- Comes from the same source as training
- Same distribution
- Same augmentation style
- Same patient demographic

**Test data:**
- Completely separate
- Slightly different distribution
- No augmentation
- Often has noisier labels and edge cases

This is the same reason Kaggle competitions show a validation–test gap.

Your models generalize well, but **real-world evaluation is always harsher**.

#### 5. Why EfficientNetV2-M Is the Best Model

Your evaluation summary ranks it #1:

> 🏆 Best Model: tf_efficientnetv2_m (89.20%)

And looking at sensitivity:
- Best sensitivity on pneumonia classes
- Best balance of precision/recall
- Best confusion matrix symmetry
- Smallest drop from validation to test

This matches EfficientNet's known strengths:
- Better inductive bias
- More stable gradients
- Better fine-tuning transfer
- Multi-scale features, which help with X-ray variability

ConvNeXt models are excellent, but more brittle when dataset is small.

#### 6. What We Can Fully Conclude From This Evaluation

Here is the **clear, correct, high-level interpretation** of the complete evaluation:

**1. All models perform well, between 78–89% accuracy.**
- No model collapsed or overfit badly

**2. EfficientNetV2-M and L consistently dominate ConvNeXt and RegNet.**
- Better generalization

**3. Two classes remain clinically hard for every model.**
- class_0 (bacterial pneumonia)
- class_4 (viral pneumonia)

Because:
- High texture similarity
- Mild symptoms often appear identical in X-ray
- Dataset is noisy and imbalanced

**4. Sensitivity warnings are expected and correct.**
- Your evaluation script is working exactly as intended

**5. ConvNeXtV2-Base loses more accuracy in evaluation because it overfits more.**
- It had the highest validation accuracy but loses almost 2.5% on the test set

**6. RegNetY models are clearly not suitable for this dataset.**
- Consistent sensitivity collapse on pneumonia classes

**7. Evaluation shows the true generalization ability, not training optimism.**
- This is the real-world performance

#### 7. What You Should Do Next (Recommended Plan)

If your goal is higher sensitivity (recall) for pneumonia subclasses:

**A. Use model ensembling**
- Combine EffNetV2-M + V2-L + ConvNeXtV2-Base

**B. Use recall-oriented training**
- Focal loss
- Label smoothing
- Class-balanced loss
- Weighted cross-entropy

**C. Increase data variance**
- CLAHE
- Histogram equalization
- Stronger augmentation
- Mixup or CutMix

**D. Improve pneumonia class annotations**
- This is the biggest bottleneck in many public X-ray datasets

#### 8. Final Summary in One Sentence

Your evaluation log shows that all models generalize well, EfficientNetV2-M performs best, and—consistently across all architectures—bacterial and viral pneumonia remain the hardest classes, causing lower sensitivity and triggering the warnings.

### Training Log Interpretation

This section provides a comprehensive guide to interpreting your training logs and understanding model performance across different architectures.

#### 1. Overall Conclusion

Across 7 models, **ConvNeXtV2 and EfficientNetV2 dominate**, with the absolute best validation accuracy achieved by **convnextv2_base (89.27%)**, but the **best test accuracy comes from EfficientNetV2-M (89.20%)**.

**Key Insight:**
- **EfficientNetV2-M generalizes better to unseen data than ConvNeXtV2-Base**, even though ConvNeXtV2-Base wins on validation.
- The differences are small but consistent: EfficientNetV2 models produce **more stable test performance**.

#### 2. Validation vs Test (Generalization Behaviour)

**Validation Leader:**
| Model               | Val Acc    |
| ------------------- | ---------- |
| **ConvNeXtV2-Base** | **89.27%** |
| ConvNeXtV2-Tiny     | 87.80%     |
| EffNetV2-L          | 86.92%     |
| EffNetV2-M          | 86.38%     |

**Test Leader:**
| Model           | Test Acc   |
| --------------- | ---------- |
| **EffNetV2-M**  | **89.20%** |
| EffNetV2-L      | 88.69%     |
| ConvNeXtV2-Tiny | 85.36%     |
| ConvNeXtV2-Base | 86.75%     |

**Key Insight:**
- ConvNeXtV2-Base has **excellent validation accuracy**, but **noticeable drop on test** (89.27% → 86.75%).
- EffNetV2-M has slightly lower validation accuracy, but **best test performance**.
- ➡️ This means **ConvNeXtV2-Base is slightly overfitting your validation set**, while **EffNetV2-M is more robust**.

#### 3. Architecture Ranking and Behavior Summary

**1️⃣ EfficientNetV2-M — Best real-world performance**
- Very stable training curves
- Extremely strong on COVID, Tuberculosis, Normal
- Does not collapse on Bacterial or Viral Pneumonia
- Test accuracy peaks at **89.20%**
- Balanced bias/variance behaviour

**2️⃣ EfficientNetV2-L — Second best**
- Higher capacity model
- Slightly better than ConvNeXtV2-Base on test
- Also very stable
- Nearly identical per-class behaviour to EfficientNet-M

**3️⃣ ConvNeXtV2-Base — Best validation, but worse test**
- Brilliant backbone, strong feature extraction
- However: small train/val gap + drop on test
- Mild overfitting to validation distribution
- Very high Normal & Tuberculosis, but weaker on Viral Pneumonia

**4️⃣ ConvNeXtV2-Tiny**
- Stronger than RegNet, weaker than EfficientNetV2-M/L
- Noticeable drop on test (from ~88 val → 85 test)

**5️⃣–7️⃣ RegNet models**
- Consistently weaker
- Low performance on Viral & Bacterial Pneumonia
- Smaller representational power
- Expected to underperform compared to modern transformers / hybrid-CNNs

#### 4. What Your Training Curves Say

**A. Stage 1: Freezing the backbone**

All models show:
- Rapid improvement in the head
- **No overfitting**
- Validation improving monotonically

This means:
- → Your head architecture is perfectly tuned
- → Image size, batch size, and augmentation are correct
- → Learning rate 1e-4 for frozen stages is appropriate

**B. Stage 2: Unfreezing the backbone**

Across all models:
- Training accuracy climbs to ~90–94%
- Validation improves early, then plateaus
- Cosine LR decreases too quickly but still works
- No divergence or instability

This is ideal behaviour:
- → Backbone low learning rate (5e-6) is correctly chosen
- → Your head LR (3e-5) is slightly higher, good
- → No sign of catastrophic forgetting

But some hints of:
- → Mild overfitting starting around epoch 8–12

You prevent this thanks to early stop and small LR.

#### 5. Per-Class Behaviour (From Test Logs)

Across all models, patterns repeat:
- **Corona Virus Disease**: easiest class (~95–97% recall)
- **Tuberculosis**: also very easy (~95–99% recall)
- **Normal**: almost always >95% recall
- **Bacterial Pneumonia**: hardest (63–77% recall)
- **Viral Pneumonia**: second hardest (60–79% recall)

This suggests:
- → Dataset difficulty is asymmetric
- → Classes with overlapping radiological appearance (bacterial/viral pneumonia) dominate errors
- → Better augmentations or class-balanced sampling may help

#### 6. Batch Size Effects

You used:
- Tiny batch sizes for large models (bs=4–8)
- Larger batch sizes for tiny models (bs=16–64)

Large models (EffNetV2-L, M) with small batch sizes:
- Still stable
- Benefit from high resolution

ConvNeXtV2:
- Performance slightly improves with larger batch sizes
- But doesn't generalize as well as EffNetV2-M/L

**Conclusion:**
- → Small batch sizes are **not hurting** EfficientNet models
- → ConvNeXtV2 might improve more with larger batch + gradient accumulation
- → This can be your next optimization

#### 7. Why EffNetV2-M Generalizes the Best

EfficientNetV2 benefits from:
- **Strong regularization** (stochastic depth, squeeze-excitation, fused MBConv)
- **Better inductive biases** than pure ConvNeXt
- Excellent scaling for medium resolution (320x320)

ConvNeXt is more modern, but:
- Transformer-like convs
- Larger models may memorize dataset structure
- Small medical datasets → EffNet family often wins

### Test Results Interpretation

This section provides a comprehensive analysis of test results across all models, revealing which models truly generalize best to unseen data.

#### 1. Final Ranking (Test Accuracy Only)

From comprehensive test evaluation:

| Rank  | Model                   | **Test Accuracy** |
| ----- | ----------------------- | ----------------- |
| **1** | **tf_efficientnetv2_m** | **89.20%**        |
| **2** | tf_efficientnetv2_l     | 88.69%            |
| **3** | convnextv2_base         | 86.75%            |
| **4** | convnextv2_tiny         | 85.36%            |
| **5** | tf_efficientnetv2_s     | 85.11%            |
| **6** | regnety_006             | 81.12%            |
| **7** | regnety_004             | 78.51%            |

**Verdict:**

**EfficientNetV2-M is your best model for real-world generalization**

It outperforms ConvNeXtV2-Base and even its bigger sibling EfficientNetV2-L.

This confirms what we suspected from training:
- ConvNeXtV2-Base overfits slightly to validation
- EfficientNetV2 models (especially M and L) generalize better

#### 2. Why EfficientNetV2-M Wins

Looking at its classification performance:

**EffNetV2-M per-class recall (from test logs):**
- **Corona Virus Disease:** 96.55%
- **Normal:** 95.15%
- **Tuberculosis:** 97.79%
- **Viral Pneumonia:** 78.12%
- **Bacterial Pneumonia:** 76.71%

This is extremely balanced:
- No collapse in pneumonia classes
- Excellent performance on all major disease types

**Key strength:**

EffNetV2-M makes **fewer extreme overconfident mistakes** compared to ConvNeXt.

This is critical for medical imaging applications.

#### 3. Why ConvNeXtV2-Base Drops on Test

ConvNeXtV2-Base test accuracy: **86.75%**  
Validation accuracy: **89.27%**

**This is a strong sign of mild overfitting.**

Look at ConvNeXt's weaknesses:
- **Viral Pneumonia recall:** 71.88%
- **Bacterial Pneumonia recall:** 77.26%
- **Corona Virus recall:** 92.86% (lower than EffNetV2-M)

ConvNeXtV2-Base is **excellent on Normal and Tuberculosis**, but struggles on the subtle pneumonia classes, which are the hardest.

EffNetV2-M handles these better → better test generalization.

#### 4. Architectural Behavior Observed in Test Logs

**EfficientNetV2 Family:**
- Very consistent results across M and L
- Extremely high precision/recall on "Normal", "COVID", "TB"
- More stable across distribution shifts
- Better inductive biases for medical texture classification
- Good with small batch sizes (8)

**ConvNeXtV2 Family:**
- Strong on training/validation
- Slight drop in unseen data
- Handles sharp edges & textures well but slightly weaker for subtle pneumonia differences
- Normal class sometimes over-predicted

**RegNet models:**
- Struggle heavily with Viral/Bacterial pneumonia
- Expected due to lower capacity
- Not ideal for this problem

#### 5. Deeper Interpretation: Class Difficulty

Across **all 7 models**, the same pattern repeats:

**Easiest classes:**
1. **Tuberculosis** (up to 99%)
2. **Corona Virus Disease** (~95–97%)
3. **Normal** (~94–97%)

**Hardest classes:**
- **Bacterial Pneumonia** (most models: 62–77%)
- **Viral Pneumonia** (most models: 70–78%)

This reveals:

**→ The dataset has overlapping visual features between:**
- Bacterial vs viral pneumonia
- Pneumonia vs normal (mild cases)

Your dataset is **not completely separable** in these classes.

This is the biological reality of chest X-ray interpretation.

EffNetV2 models handle these subtleties better.

#### 6. What You Can Learn From All Logs Together

**1. EfficientNetV2-M is your best model.**
- Highest test accuracy + most stable behaviour

**2. ConvNeXtV2-Base slightly overfits.**
- Even though it wins on validation

**3. Bacterial/Viral pneumonia are the bottleneck.**
- Nearly all misclassification comes from confusion between these two

**4. Your augmentation pipeline is good.**
- None of the models show chaotic behavior

**5. Training pipeline is stable**
- No exploding gradients
- Cosine LR schedule works
- Two-stage training works
- LR ratios between head/backbone are correct

**6. Dataset test distribution differs slightly from validation**
- This is expected for medical datasets

#### 7. What You Should Do Next

These are the **highest-impact improvements** toward breaking **90–91%+** test accuracy.

**A. Use an Ensemble of:**
1. EfficientNetV2-M
2. ConvNeXtV2-Base

Soft-voting or average softmax.

**Expected improvement: +1–2% test accuracy.**

**B. Add class-balanced sampling**
- Especially for pneumonia classes

**C. Use Mixup (0.1) + Cutmix (0.2)**
- EffNet models respond very well to these

**D. Train longer with cosine schedule**
- Increase T_max to **30 or 45**
- Your cosine schedule decays too fast

**E. Try image size 384 for ConvNeXtV2-Base**
- The model will improve likely **+0.5–1.0%** on pneumonia classes

#### 8. Final Summary (Short & Clear)

- **Best real-world model:** EfficientNetV2-M — 89.20% (Test)
- **Best validation model:** ConvNeXtV2-Base — 89.27% (Val)
- **Most balanced class performance:** EfficientNetV2-M
- **Weakest classes across all models:** Bacterial & Viral Pneumonia
- **Strongest classes across all models:** TB, COVID, Normal
- **Pipeline is stable → improvements now rely on augmentations + ensembles + LR tuning**

#### 9. Why Test Results Differ from Training/Validation

**Yes — it is absolutely normal that test results differ from training/validation results**, and it happens even in perfectly healthy ML pipelines.

Your logs show a *typical* and *expected* pattern for real-world datasets, especially **medical imaging**.

**1. It is normal — and expected — to see differences between val and test**

You observed things like:
- **ConvNeXtV2-Base:** Val: 89.27% → Test: 86.75%
- **EfficientNetV2-M:** Val: 86.38% → Test: 89.20%

This happens in every serious project.

The test set represents a **different distribution** from training/validation, so models shift slightly.

**2. The test distribution *is* slightly different (confirmed by your logs)**

Your test log shows class counts:
```
True label distribution: [365, 406, 392, 407, 384]  ← Test data
```

Your train/val split is **not identical**:
- The test set has a slightly different ratio of Normal / TB / Pneumonia
- The test set is 1954 images — larger and more diverse
- The pneumonia classes were more balanced in val, but more varied in test

This small difference is enough to cause:
- Some models to improve (EffNetV2-M generalizes better)
- Some models to drop (ConvNeXtV2-Base slightly overfits the val distribution)

This behaviour is perfectly normal.

**3. The validation set is *not the same distribution* as the test set**

This happens because:

**A. Your validation set comes from the same split as training**

Even if random, the images in val and train share:
- Same hospitals
- Same devices
- Same patient demographics
- Identical augmentations
- Similar conditions

This makes validation slightly "optimistic."

**B. The test set is more independent**

Your test set may include:
- Different imaging conditions
- More diverse pneumonia cases
- Different brightness/contrast patterns
- Slightly different disease severity distribution
- Different class balance

This makes the test set "more realistic."

So, the test performance is **usually lower** or **shifts slightly**, depending on the model.

EfficientNetV2-M *benefits* from this.

ConvNeXtV2-Base *suffers* from this.

**4. Some architectures generalize better than others**

EfficientNetV2 models are designed with:
- Heavy regularization
- Squeeze-and-excitation
- Better inductive biases for texture
- Better resistance to overfitting
- Better performance with small batch sizes

ConvNeXtV2 models:
- Are more modern and very strong
- But behave slightly more like "transformer-like conv backbones"
- Which can overfit subtle frequency patterns

This explains why:
- **Val winner** → ConvNeXtV2-Base
- **Test winner** → EfficientNetV2-M

Again: totally normal.

**5. Small differences in image size also cause shifts**

Your models use different input sizes:
- 384×384 (EffNetV2-L)
- 320×320 (EffNetV2-M)
- 288×288 (EffNetV2-S)
- 224×224 (ConvNeXt)

The test set is **resized** accordingly, but:
- Details are lost differently
- Pneumonia textures become more or less visible
- Noise is treated differently

This changes each model's accuracy in predictable ways.

EffNetV2-M at 320×320 happens to hit the sweet spot.

**6. Pneumonia classes are biologically ambiguous**

Your hardest classes are always the same:
- Bacterial Pneumonia
- Viral Pneumonia

Across all models, these two are systematically ~10–20% worse than:
- Normal
- COVID
- TB

This is not a mistake — their radiological patterns overlap in real medicine.

So even a perfect model will score differently depending on:
- The mix of pneumonia cases in test
- The specific severe or mild cases in test
- Contrast characteristics of the scan

If test has more "borderline pneumonia," accuracy changes.

**7. Summary — the difference is normal and expected**

**Why the test differs from train/val:**

1. **Different data distribution** — more realistic and broader variation
2. **Class ratios differ** — slightly more of the harder classes
3. **Pneumonia is inherently ambiguous** — classification difficulty changes with sample composition
4. **Architectures generalize differently** — EfficientNetV2 is better at generalizing
5. **Validation is "optimistic"** — same split as training
6. **Test is "neutral"** — independent, more variable sample

**What it means:**

- Your pipeline is correct
- Your models behaved exactly as expected
- The differences are not a problem — they are useful
- EfficientNetV2-M is the most robust architecture in your experiment

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

#### Volume Mount Strategy

The Docker image contains the `projects/` directory at `/workspace/projects`. To preserve access to both the image's projects and your host files, mount your home directory to `/workspace/host` instead of `/workspace`.

**Directory Structure Inside Container:**
- `/workspace/projects` → Projects from Dockerfile image (always available)
- `/workspace/host` → Your host home directory (mounted from `$HOME`)

#### Running the Container

**Option 1: Using the Run Script (Recommended)**

From the repository root:

```bash
./run_container.sh
```

**Option 2: Manual Docker Run**

Run the training in a Docker container with GPU support:

```bash
docker run -it --runtime=nvidia \
  --entrypoint bash \
  -e KAGGLE_USERNAME=$KAGGLE_USERNAME \
  -e KAGGLE_KEY=$KAGGLE_KEY \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e HOME=/workspace \
  -v $HOME:/workspace/host \
  ghcr.io/behnamasadi/kaggle-projects:latest
```

**Alternative: Using --gpus all**

If your system supports it:

```bash
docker run -it --gpus all \
  --entrypoint bash \
  -e KAGGLE_USERNAME=$KAGGLE_USERNAME \
  -e KAGGLE_KEY=$KAGGLE_KEY \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e HOME=/workspace \
  -v $HOME:/workspace/host \
  ghcr.io/behnamasadi/kaggle-projects:latest
```

#### Running Training Inside Container

Once inside the container, navigate to the project and run training:

```bash
# Access the project from the Docker image
cd /workspace/projects/Lung_Disease_Dataset

# Run training
python scripts/train.py --config configs/train_runpod.yaml
```

**Note:** The project is already available at `/workspace/projects/Lung_Disease_Dataset` from the Docker image. You don't need to copy it from your host.

#### Accessing Host Files

If you need to access files from your host machine:

```bash
# Your host files are available at /workspace/host
ls /workspace/host

# Copy files from host to project
cp /workspace/host/my_data.csv /workspace/projects/Lung_Disease_Dataset/data/
```

#### Verifying Docker Image Contents

After building your Docker image, you can verify the projects directory size inside the container:

```bash
# Run container without mounts to see image contents
docker run -it --entrypoint bash ghcr.io/behnamasadi/kaggle-projects:latest

# Inside the container, check the size of the projects directory
cd /workspace/projects
du -sh .  # Human-readable size of the current directory
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
