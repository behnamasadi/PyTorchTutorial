# Configuration Files Guide

This directory contains configuration files organized following ML best practices.

## Configuration Structure

### **data.yaml** - Shared Data Configuration
Contains data-related settings used across training, evaluation, and inference:
- Dataset paths (train/val/test)
- Data characteristics (num_classes, class_names, image_size)
- DataLoader defaults
- Normalization constants
- Augmentation settings (reference)

**Usage:** Load this first, then override in train/eval configs as needed.

### **model.yaml** - Model Architecture Configuration
Contains model-specific settings that don't change between training and evaluation:
- Available model architectures
- Model hyperparameters
- Classifier head configuration
- Default model selection

**Usage:** Defines all available models and their properties.

### **train.yaml** - Training Configuration
Training-specific settings:
- Model selection
- Training strategy (two-stage medical fine-tuning)
- Learning rates, epochs, schedulers
- Optimizer settings
- Training data paths (references data.yaml)
- Monitoring/logging settings

**Usage:** Complete training configuration. References data.yaml and model.yaml.

### **train_local.yaml** - Local Development Training Configuration
Environment-specific training config for local machines (smaller GPUs):
- Uses `model.yaml` batch_size defaults (small values, e.g., 4-16)
- Lower `num_workers` (typically 4)
- Optimized for local development

**Usage:** `python scripts/train.py --config configs/train_local.yaml`

### **train_runpod.yaml** - RunPod Training Configuration
Environment-specific training config for RunPod (larger GPUs):
- Overrides `model.yaml` with larger batch sizes (e.g., 32+)
- Higher `num_workers` (typically 8)
- Optimized for cloud GPU instances

**Usage:** `python scripts/train.py --config configs/train_runpod.yaml`

### **eval.yaml** - Evaluation Configuration
Evaluation-specific settings:
- Model checkpoint path
- Evaluation split (test/val/train)
- Metrics to compute
- Evaluation data paths (references data.yaml)
- Output/visualization settings

**Usage:** Complete evaluation configuration. References data.yaml and model.yaml.

## Best Practices

### 1. **Avoid Duplication - Use References**
Instead of duplicating data paths, reference them:

```yaml
# ❌ BAD - Duplicated in each file
# train.yaml
data:
  path: "./data/train"
  
# eval.yaml
data:
  path: "./data/test"

# ✅ GOOD - Single source of truth
# data.yaml
paths:
  train: "./data/train"
  test: "./data/test"

# train.yaml (references data.yaml)
data:
  path: "./data/train"  # Or use: ${data.paths.train}
```

### 2. **Separate Concerns**
- **Model config**: Architecture, doesn't change
- **Training config**: Training-specific, changes per experiment
- **Eval config**: Eval-specific, changes per evaluation
- **Data config**: Shared data settings

### 3. **Configuration Hierarchy**
```
data.yaml (base)
    ↑
    ├── train.yaml (extends data.yaml)
    └── eval.yaml (extends data.yaml)

model.yaml (base)
    ↑
    ├── train.yaml (uses model.yaml)
    └── eval.yaml (uses model.yaml)
```

### 4. **Load Configuration Pattern**

```python
import yaml
from pathlib import Path

def load_config(config_name: str):
    """Load configuration with inheritance."""
    # Base configs
    data_config = yaml.safe_load(open("configs/data.yaml"))
    model_config = yaml.safe_load(open("configs/model.yaml"))
    
    # Task-specific config
    task_config = yaml.safe_load(open(f"configs/{config_name}.yaml"))
    
    # Merge (task config overrides base configs)
    config = {
        **data_config,
        **model_config,
        **task_config
    }
    
    return config
```

### 5. **Data Configuration Principles**

**What goes in data.yaml:**
- Dataset paths (single source of truth)
- Data characteristics (num_classes, class_names)
- DataLoader defaults
- Normalization constants
- Shared augmentation settings

**What goes in train.yaml (data section):**
- Training-specific overrides (batch_size, shuffle, augmentation)
- Training data path (reference or override)
- Validation data path

**What goes in eval.yaml (data section):**
- Evaluation-specific overrides (batch_size, shuffle=false)
- Test/val data path (reference or override)
- No augmentation settings

## Example Usage

### Training (Default)
```bash
python scripts/train.py --config configs/train.yaml
```

### Training (Local Development)
```bash
python scripts/train.py --config configs/train_local.yaml
```
Uses smaller batch sizes from `model.yaml` defaults, suitable for local GPUs.

### Training (RunPod/Cloud)
```bash
python scripts/train.py --config configs/train_runpod.yaml
```
Uses larger batch sizes (overrides `model.yaml`), suitable for cloud GPUs with more VRAM.

The training script will:
1. Load `data.yaml` for data defaults
2. Load `model.yaml` for model architecture
3. Load the specified train config (`train.yaml`, `train_local.yaml`, or `train_runpod.yaml`) and merge/override
4. Use merged config for training

### Batch Size Resolution Priority
1. `train_*.yaml` `data.batch_size` (if specified) - **highest priority**
2. `model.yaml` `models.{model_name}.batch_size` (model-specific defaults)
3. `data.yaml` `dataloader.batch_size` (fallback default: 32)

**Strategy:**
- `model.yaml` contains small batch sizes suitable for local development
- `train_local.yaml` omits `batch_size` to use `model.yaml` defaults
- `train_runpod.yaml` explicitly sets larger `batch_size` to override `model.yaml`

### Evaluation
```bash
python scripts/evaluate.py --config configs/eval.yaml --checkpoint ./checkpoints/best_model.pt
```

The evaluation script will:
1. Load `data.yaml` for data defaults
2. Load `model.yaml` for model architecture
3. Load `eval.yaml` and merge/override
4. Use merged config for evaluation

## Updating Configurations

### Update Data Paths
Edit only `data.yaml` - changes propagate to all configs that reference it.

### Update Model Architecture
Edit `model.yaml` - affects all training and evaluation.

### Update Training Strategy
Edit `train.yaml` - affects only training runs.

### Update Evaluation Settings
Edit `eval.yaml` - affects only evaluation runs.

## Advanced: Using Hydra (Optional)

For more sophisticated configuration management, consider using Hydra:

```yaml
# configs/train.yaml
defaults:
  - data: data
  - model: model
  - _self_

# Training-specific overrides
training:
  epochs: 50
```

This automatically handles configuration composition and overrides.

