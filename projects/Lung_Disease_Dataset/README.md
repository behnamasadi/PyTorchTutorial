# Brain Cancer MRI Classification

A comprehensive deep learning project for "Lungs Disease Dataset" classification using chest X-Rays images, featuring state-of-the-art architectures, medical AI validation.

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)


## Project Overview

This project implements a various convolutional neural network to classify chest X-Rays into the following categories:

- **Viral Pneumonia**
- **Bacterial Pneumonia**
- **Covid**
- **Tuberculosis**
- **Normal**

[Dataset on Kaggle](https://www.kaggle.com/datasets/omkarmanohardalvi/lungs-disease-dataset-4-types)


```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("omkarmanohardalvi/lungs-disease-dataset-4-types")
```


or


```python
kaggle datasets list -s "omkarmanohardalvi/lungs-disease-dataset-4-types"
kaggle datasets download "omkarmanohardalvi/lungs-disease-dataset-4-types"
```



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

### **Medical Fine-Tuning Strategy**

Since medical images (chest X-rays) are fundamentally different from ImageNet natural images, this project uses a **two-stage medical fine-tuning policy** - the most standard approach in medical imaging:

#### **Stage 1 - Freeze Backbone, Train Classifier Only**

```python
# Freeze the entire backbone
freeze(backbone)

# Train only the classifier head
train(classifier_head)

# Training parameters:
# - Epochs: 3-10
# - Learning rate: 1e-3 or 1e-4
```

**Purpose:** 
- Initialize the classifier with task-specific features
- Prevent early overfitting to medical domain specifics
- Stabilize training before fine-tuning the backbone

#### **Stage 2 - Unfreeze Backbone, End-to-End Training**

```python
# Unfreeze the entire model
unfreeze(backbone)

# Train entire model end-to-end
train(entire_model)

# Training parameters:
# - Epochs: 10-50
# - Learning rate: 1e-5 or 3e-5 (much lower!)
# - Learning rate schedule: Cosine annealing or ReduceLROnPlateau
```

**Purpose:**
- Adapt pre-trained ImageNet features to medical domain
- Fine-tune low-level features (edges, textures) for medical patterns
- Achieve better domain-specific feature representation

**Why Two-Stage Training?**

Medical images (X-rays, CT scans, MRIs) have fundamentally different characteristics than natural images:
- **Different textures**: Medical images have distinct patterns (bone structures, soft tissues, anomalies)
- **Different semantics**: Features learned on ImageNet (cats, cars, objects) need adaptation
- **Different scales**: Medical anomalies can be subtle and require fine-grained feature learning
- **Domain gap**: Large gap between natural and medical images requires gradual adaptation

This two-stage approach is the **universal "medical fine-tuning" recipe** used in:
- Medical imaging research papers
- Clinical deployment pipelines
- FDA-approved medical AI systems

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
