# Kaggle Dataset Downloader

Automated script to download Kaggle datasets and create symlinks to your project data directory.

## What This Does

This script:
1. Reads dataset configuration from `config/train.yaml`
2. Downloads the specified Kaggle dataset using `kagglehub` (stores in cache: `~/.cache/kagglehub/`)
3. Creates a symlink from `kaggle_structure/data/train` to the downloaded dataset
4. Works both locally (using `kaggle.json`) and on RunPod (using environment variables)

**Benefits:**
- No duplicate data - uses symlinks
- Fast downloads - leverages Kaggle's cachedata
- Works seamlessly in local and cloud environments

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Usage](#usage)
4. [Finding Kaggle Datasets](#finding-kaggle-datasets)
5. [Example Datasets](#example-datasets)
6. [Running in Kaggle Notebooks](#running-in-kaggle-notebooks)
7. [Kaggle Hardware & GPU](#kaggle-hardware--gpu)
8. [GPU Optimization Best Practices](#gpu-optimization-best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## Installation

### Prerequisites

```bash
pip install kagglehub pyyaml
```

### Kaggle Authentication

You need Kaggle API credentials to download datasets.

#### Option 1: Local Development (kaggle.json)

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings/account)
2. Scroll to "API" section
3. Click "Create New Token"
4. Download `kaggle.json`
5. Place it in the correct location:

**Linux/macOS:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Windows:**
```powershell
# Place kaggle.json in:
C:\Users\<YourUsername>\.kaggle\kaggle.json
```

#### Option 2: RunPod/Docker (Environment Variables)

Set environment variables in your RunPod/Docker container:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

Or add to your Dockerfile:
```dockerfile
ENV KAGGLE_USERNAME=your_username
ENV KAGGLE_KEY=your_api_key
```

### kaggle.json Locations Reference

| Platform | Location |
|----------|----------|
| **Linux** | `~/.kaggle/kaggle.json` or `~/.config/kaggle/kaggle.json` |
| **macOS** | `~/.kaggle/kaggle.json` |
| **Windows** | `C:\Users\<YourUsername>\.kaggle\kaggle.json` |
| **WSL** | `/home/<username>/.kaggle/kaggle.json` |

---

## Configuration

Edit `config/train.yaml`:

```yaml
data:
  kaggle_dataset: "owner/dataset-name"  # e.g., "masoudnickparvar/brain-tumor-mri-dataset"
  path: "./data/train"                  # Relative to kaggle_structure directory
```

---

## Usage

### Running the Script

```bash
# From the PyTorchTutorial environment root
python src/kaggle_structure/scripts/index.py
```

### Expected Output

```
kagglehub is available: 0.3.12
Reading config from /path/to/kaggle_structure/config/train.yaml
masoudnickparvar/brain-tumor-mri-dataset
./data/train

Attempting to download dataset: masoudnickparvar/brain-tumor-mri-dataset

======================================================================
✅ Dataset downloaded successfully
======================================================================
  Source (Kaggle cache): /home/user/.cache/kagglehub/datasets/.../versions/2
  Target (Project dir):  /path/to/kaggle_structure/data/train
======================================================================

✅ Symlink created successfully
   /path/to/kaggle_structure/data/train -> /home/user/.cache/kagglehub/.../versions/2
   (no disk space wasted)
======================================================================
```

### Project Structure

```
kaggle_structure/
├── config/
│   └── train.yaml          # Dataset configuration
├── scripts/
│   └── index.py            # Download script
├── data/                   # Created by script
│   └── train/              # Symlink to Kaggle cache
└── README.md               # This file
```

---

## Finding Kaggle Datasets

### Method 1: Kaggle Website

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Search for your topic with relevant keywords:
   - **Medical:** "Brain MRI", "Tumor", "Cancer", "Lung", "Histopathology", "medical imaging"
   - **Computer Vision:** "object detection", "segmentation", "saliency", "image similarity"
   - **3D Vision:** "LiDAR", "SLAM", "Photogrammetry", "stereo", "monocular depth"
3. Click on a dataset
4. The dataset name is in the URL: `kaggle.com/datasets/OWNER/DATASET-NAME`
5. Use `OWNER/DATASET-NAME` in your `config/train.yaml`

**Example searches:**
- "Brain Tumor MRI" → Brain tumor detection datasets
- "lung cancer CT" → Lung cancer imaging datasets
- "object detection COCO" → Object detection datasets
- "LiDAR point cloud" → 3D LiDAR datasets
- "histopathology cancer" → Medical histopathology images
- "semantic segmentation" → Segmentation datasets
- "SLAM dataset" → Robotics SLAM datasets

### Method 2: Kaggle CLI

The Kaggle CLI is a powerful tool for searching and downloading datasets from the command line.

#### Installation

```bash
pip install kaggle
```

#### Searching Datasets

**Basic search:**
```bash
# Search by keyword
kaggle datasets list -s "MRI"
kaggle datasets list -s "Brain Tumor"
kaggle datasets list -s "object detection"
```

**Sort options:**

Available sort methods: `hottest`, `votes`, `updated`, `active`, `published`

```bash
# Sort by votes (most popular)
kaggle datasets list -s "MRI" --sort-by votes

# Sort by recently updated
kaggle datasets list -s "Brain Tumor" --sort-by updated

# Sort by most active (recently discussed)
kaggle datasets list -s "lung cancer" --sort-by active

# Sort by hottest (trending)
kaggle datasets list -s "histopathology" --sort-by hottest
```

**Pagination:**

Use `-p` or `--page` to see more results:

```bash
# Get page 1 (default, shows first 20 results)
kaggle datasets list -s "MRI"

# Get page 2
kaggle datasets list -s "MRI" --sort-by updated -p 2

# Get page 3 with specific sorting
kaggle datasets list -s "medical imaging" --sort-by votes -p 3
```

**More search options:**

```bash
# Show more results per page (max 100)
kaggle datasets list -s "segmentation" --max-size 50

# Search by size (small, medium, large)
kaggle datasets list -s "brain" --file-type csv

# Search by license
kaggle datasets list -s "MRI" --license cc
```

**Output format:**

```bash
# Example output:
# masoudnickparvar/brain-tumor-mri-dataset
# navoneel/brain-mri-images-for-brain-tumor-detection
# fernando2rad/brain-tumor-mri-images-44c
```

Copy the dataset identifier (e.g., `masoudnickparvar/brain-tumor-mri-dataset`) and use it in your `config/train.yaml`.

#### Downloading Datasets

**Download directly with CLI:**

```bash
# Download a dataset
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset

# Download to specific directory
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p ./data/

# Download without unzipping
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset --unzip false

# Force re-download
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset --force
```

**Get dataset information:**

```bash
# Get dataset metadata
kaggle datasets metadata masoudnickparvar/brain-tumor-mri-dataset

# List files in a dataset (without downloading)
kaggle datasets files masoudnickparvar/brain-tumor-mri-dataset
```

#### Quick Workflow

```bash
# 1. Search for datasets
kaggle datasets list -s "Brain Tumor" --sort-by votes

# 2. Pick a dataset (e.g., masoudnickparvar/brain-tumor-mri-dataset)

# 3. Check what's in it
kaggle datasets files masoudnickparvar/brain-tumor-mri-dataset

# 4. Add to your config/train.yaml
# data:
#   kaggle_dataset: "masoudnickparvar/brain-tumor-mri-dataset"
#   path: "./data/train"

# 5. Run your download script
python src/kaggle_structure/scripts/index.py
```

#### Useful CLI Commands Reference

| Command | Description | Example |
|---------|-------------|---------|
| `kaggle datasets list -s "keyword"` | Search datasets | `kaggle datasets list -s "MRI"` |
| `--sort-by [method]` | Sort results | `--sort-by votes` |
| `-p [page]` | Pagination | `-p 2` |
| `kaggle datasets download -d [dataset]` | Download dataset | `kaggle datasets download -d owner/dataset` |
| `kaggle datasets files [dataset]` | List files without downloading | `kaggle datasets files owner/dataset` |
| `kaggle datasets metadata [dataset]` | Get dataset info | `kaggle datasets metadata owner/dataset` |

**Sort methods:**
- `hottest` - Trending datasets (default)
- `votes` - Most upvoted (most popular)
- `updated` - Recently updated
- `active` - Recently discussed
- `published` - Recently published

### Method 3: Browse by Category

Visit specific categories:
- **Sports:** https://www.kaggle.com/datasets?topic=sports
- **Medicine:** https://www.kaggle.com/datasets?topic=health
- **Computer Vision:** https://www.kaggle.com/datasets?topic=computer-vision
- **NLP:** https://www.kaggle.com/datasets?topic=nlp

---

## Example Datasets

Try these popular datasets organized by research area:

### Medical Imaging

```yaml
# Brain MRI - Tumor Detection
kaggle_dataset: "masoudnickparvar/brain-tumor-mri-dataset"

# Brain MRI - Segmentation
kaggle_dataset: "mateuszbuda/lgg-mri-segmentation"

# Brain Cancer MRI
kaggle_dataset: "orvile/brain-cancer-mri-dataset"

# Lung Cancer Detection
kaggle_dataset: "mohamedhanyyy/chest-ctscan-images"

# Lung X-Ray Segmentation
kaggle_dataset: "nikhilpandey360/chest-xray-masks-and-labels"

# Histopathology - Cancer Detection
kaggle_dataset: "andrewmvd/lung-and-colon-cancer-histopathological-images"

# Chest X-Ray Pneumonia
kaggle_dataset: "paultimothymooney/chest-xray-pneumonia"

# Medical Image Segmentation
kaggle_dataset: "kmader/finding-lungs-in-ct-data"
```

### Computer Vision

```yaml
# Object Detection - COCO
kaggle_dataset: "awsaf49/coco-2017-dataset"

# Object Detection - Pascal VOC
kaggle_dataset: "aladdinpersson/pascal-voc-dataset-used-in-yolov3-video"

# Image Similarity / Saliency
kaggle_dataset: "adrianmcmahon/image-similarity"

# Semantic Segmentation - Cityscapes
kaggle_dataset: "dansbecker/cityscapes-image-pairs"

# LiDAR Point Cloud Data
kaggle_dataset: "sshikamaru/lidar-3d-object-detection"

# LiDAR - KITTI Dataset
kaggle_dataset: "klemenko/kitti-dataset"
```

### 3D Vision & Robotics

```yaml
# SLAM - Indoor Scenes
kaggle_dataset: "balraj98/deepslam-dataset"

# Photogrammetry - 3D Reconstruction
kaggle_dataset: "balraj98/3d-reconstruction-dataset"

# Stereo Vision
kaggle_dataset: "samfc10/middlebury-stereo-datasets"

# Monocular Depth Estimation
kaggle_dataset: "kmader/kitti-depth-completion"
```

### Quick Start Examples

If you're just testing the download script:

```yaml
# Small dataset - Brain MRI (fastest download)
kaggle_dataset: "navoneel/brain-mri-images-for-brain-tumor-detection"

# Medium dataset - Lung Cancer
kaggle_dataset: "adityamahimkar/iqothnccd-lung-cancer-dataset"

# Larger dataset - Histopathology
kaggle_dataset: "andrewmvd/lung-and-colon-cancer-histopathological-images"
```

---

## Running in Kaggle Notebooks

When running your code in a Kaggle notebook, the environment and paths are different from local/RunPod setups. Here's how to handle it properly.

### Environment Detection

Kaggle notebooks have specific paths and don't require authentication:

```python
import os
import kagglehub

def is_kaggle_notebook():
    """Detect if code is running in a Kaggle notebook."""
    return os.path.exists('/kaggle/input')

def get_data_path(dataset_name, fallback_local_path="./data"):
    """
    Get dataset path that works in Kaggle notebooks and locally.
    
    Args:
        dataset_name: Kaggle dataset identifier (e.g., "owner/dataset-name")
        fallback_local_path: Local path if download fails
    
    Returns:
        Path to dataset directory
    """
    if is_kaggle_notebook():
        # In Kaggle notebooks, datasets are pre-mounted at /kaggle/input
        # Format: /kaggle/input/dataset-name (without owner prefix)
        dataset_slug = dataset_name.split('/')[-1]  # Extract "dataset-name" from "owner/dataset-name"
        kaggle_input_path = f"/kaggle/input/{dataset_slug}"
        
        if os.path.exists(kaggle_input_path):
            print(f"✅ Running in Kaggle notebook")
            print(f"   Using pre-mounted dataset: {kaggle_input_path}")
            return kaggle_input_path
        else:
            print(f"⚠️  Dataset not found in /kaggle/input")
            print(f"   Available datasets: {os.listdir('/kaggle/input')}")
            raise FileNotFoundError(f"Dataset {dataset_slug} not found in Kaggle input")
    
    # Local or RunPod environment - download with kagglehub
    try:
        print(f"📥 Downloading dataset: {dataset_name}")
        path = kagglehub.dataset_download(dataset_name)
        print(f"✅ Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print(f"   Using fallback path: {fallback_local_path}")
        return fallback_local_path
```

### Complete Notebook Pattern

Here's a robust pattern for notebooks that work everywhere (Kaggle, local, RunPod):

```python
import os
import kagglehub
from pathlib import Path

# ============================================================================
# STEP 1: Environment Detection & Data Setup
# ============================================================================

print("\n" + "="*70)
print("STEP 1: Setting up dataset")
print("="*70)

# Configuration
DATASET_NAME = "masoudnickparvar/brain-tumor-mri-dataset"
LOCAL_FALLBACK = "./data/brain-tumor"
SUBDIRECTORY = "path/to/subdirectory"  # Dataset-specific subdirectory (if needed)

# Detect environment
IS_KAGGLE = os.path.exists('/kaggle/input')
print(f"Environment: {'Kaggle Notebook' if IS_KAGGLE else 'Local/RunPod'}")

# Get data path
data_path = get_data_path(DATASET_NAME, LOCAL_FALLBACK)

# Navigate to subdirectory if needed
if SUBDIRECTORY:
    data_path = os.path.join(data_path, SUBDIRECTORY)

# Verify final path
if os.path.exists(data_path):
    print(f"✅ Data path verified: {data_path}")
    contents = os.listdir(data_path)
    print(f"   Found {len(contents)} items")
else:
    print(f"❌ Data path not found: {data_path}")

print("="*70 + "\n")
```

### Key Differences: Kaggle vs Local

| Aspect | Kaggle Notebook | Local/RunPod |
|--------|----------------|--------------|
| **Authentication** | Not required (auto-authenticated) | Requires kaggle.json or env vars |
| **Data location** | `/kaggle/input/dataset-name` | Downloaded to `~/.cache/kagglehub/` |
| **How to add dataset** | Click "Add Data" button in notebook | Run download script |
| **Path format** | `/kaggle/input/<dataset-slug>` | `~/.cache/kagglehub/datasets/<owner>/<dataset>/<version>` |
| **Persistence** | Read-only, session-specific | Persistent across runs |
| **Output** | Save to `/kaggle/working/` | Save anywhere |

---

## Kaggle Hardware & GPU

When running code in Kaggle notebooks or competitions, be aware of hardware constraints.

### Available GPUs

| GPU | VRAM | Compute Capability | Best For |
|-----|------|-------------------|----------|
| **NVIDIA Tesla P100** | 16 GB | 6.0 | General deep learning, FP32 |
| **NVIDIA Tesla T4** | 16 GB | 7.5 | Mixed precision (FP16), inference |

> **Note:** You **cannot choose** which GPU you get. Kaggle assigns P100 or T4 randomly based on availability. Both have **16GB VRAM** - design your code for this limit.

### Checking Which GPU You Got

```python
import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {gpu_props.total_memory / 1e9:.1f} GB")
    print(f"Compute Capability: {gpu_props.major}.{gpu_props.minor}")
    
    if "P100" in gpu_name:
        print("→ Good for FP32 training")
    elif "T4" in gpu_name:
        print("→ Excellent for FP16/mixed precision (3-4x faster)")
else:
    print("No GPU available")
```

Or use:
```bash
!nvidia-smi
```

### GPU Comparison

| Aspect | P100 | T4 |
|--------|------|-----|
| **FP32 performance** | 9.3 TFLOPS (better) | 8.1 TFLOPS |
| **FP16 performance** | 18.7 TFLOPS | 65 TFLOPS (3-4x faster!) |
| **Memory bandwidth** | 732 GB/s (better) | 320 GB/s |
| **Power** | 250W | 70W (more efficient) |

**Bottom line:** Use mixed precision (FP16) - T4 will be much faster, P100 still works fine.

### System Resources

| Resource | Limit |
|----------|-------|
| **GPU Time** | ~30 hours/week |
| **CPU Cores** | 4 cores |
| **RAM** | 16-30 GB |
| **Disk Space** | ~73 GB (temporary) |
| **Session Time** | 9-12 hours max |
| **Internet** | Enabled (disabled in competitions) |

### Critical Limitations

#### 1. 16 GB VRAM (Most Common Bottleneck)

```python
# ❌ Will likely fail
model = HugeTransformer(layers=48, hidden=4096)
batch_size = 64

# ✅ Optimized for Kaggle
model = SmallerModel(layers=12, hidden=768)
batch_size = 16
```

#### 2. Session Timeout (9-12 hours)

Always save checkpoints:

```python
from pathlib import Path
import torch

checkpoint_dir = Path("/kaggle/working/checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Save during training
if epoch % save_freq == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')

# Resume from checkpoint
if (checkpoint_dir / 'latest.pth').exists():
    checkpoint = torch.load(checkpoint_dir / 'latest.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
```

#### 3. Disk Space (~73 GB)

```python
# Check space
!df -h

# Clean up when done
import shutil
shutil.rmtree('./cache', ignore_errors=True)
```

### System Specs Checker

Add this to your notebooks:

```python
import torch
import psutil

def print_system_specs():
    """Print Kaggle system specifications."""
    print("="*70)
    print("SYSTEM SPECIFICATIONS")
    print("="*70)
    
    # CPU & RAM
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    
    # GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("GPU: Not available")
    
    # Disk
    disk = psutil.disk_usage('/kaggle/working' if os.path.exists('/kaggle/working') else '.')
    print(f"Disk Space: {disk.total / 1e9:.1f} GB total, {disk.free / 1e9:.1f} GB free")
    
    print("="*70)

print_system_specs()
```

---

## GPU Optimization Best Practices

Strategies to maximize GPU utilization and avoid out-of-memory errors on Kaggle's 16GB GPUs.

### 1. Use Mixed Precision Training (FP16)

**Saves ~50% memory** and makes T4 3-4x faster:

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = YourModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Automatic mixed precision
        with autocast():
            outputs = model(batch)
            loss = criterion(outputs, targets)
        
        # Scaled backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**Memory savings:**
- FP32: 4 bytes/parameter → Model with 1B params = 4GB
- FP16: 2 bytes/parameter → Model with 1B params = 2GB
- **Result: 2x model capacity or 2x batch size**

### 2. Gradient Accumulation

Simulate larger batch sizes without using more memory:

```python
# Effective batch size = batch_size * accumulation_steps
batch_size = 8          # What fits in 16GB VRAM
accumulation_steps = 4  # Effective batch size = 32

optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    # Update weights every N steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Benefits:**
- Train with effective batch size of 32-64 on 16GB GPU
- Better gradient estimates without OOM errors
- Essential for large models

### 3. Monitor GPU Memory

Always track memory usage:

```python
def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# Use throughout your code
model = create_model()
print_gpu_memory()  # Check after model creation

# During training
for epoch in range(epochs):
    train_one_epoch()
    print_gpu_memory()  # Monitor each epoch
```

Real-time monitoring:

```bash
# Run in separate terminal or notebook cell
!nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 5
```

### 4. Find Optimal Batch Size Dynamically

Binary search for maximum batch size:

```python
def find_optimal_batch_size(model, sample_input, max_batch_size=256, min_batch_size=1):
    """
    Find largest batch size that fits in GPU memory.
    
    Args:
        model: Your PyTorch model
        sample_input: A single example input tensor
        max_batch_size: Starting point for binary search
        min_batch_size: Minimum batch size to try
    
    Returns:
        Optimal batch size
    """
    import torch
    
    model.eval()
    batch_size = max_batch_size
    
    while batch_size >= min_batch_size:
        try:
            # Create batch
            batch = sample_input.repeat(batch_size, *([1] * (sample_input.dim() - 1)))
            
            # Try forward + backward pass
            with torch.enable_grad():
                output = model(batch)
                loss = output.sum()  # Dummy loss
                loss.backward()
            
            # Success!
            torch.cuda.empty_cache()
            print(f"✅ Batch size {batch_size} fits in memory")
            return batch_size
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                batch_size //= 2  # Binary search
            else:
                raise e
    
    return min_batch_size

# Usage
optimal_bs = find_optimal_batch_size(model, sample_input)
print(f"Using batch size: {optimal_bs}")
```

### 5. Clear Unused Variables & Cache

Free memory immediately after use:

```python
# Delete large tensors when done
del large_activation_maps
torch.cuda.empty_cache()

# Use no_grad for inference
with torch.no_grad():
    predictions = model(test_data)

# Clear gradients when not needed
model.zero_grad(set_to_none=True)  # Frees memory vs zero_grad()
```

### 6. Gradient Checkpointing

Trade compute for memory (recompute activations during backward):

```python
import torch.utils.checkpoint as checkpoint

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1024, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 10)
    
    def forward(self, x):
        # Checkpoint expensive layers
        x = checkpoint.checkpoint(self.layer1, x)
        x = checkpoint.checkpoint(self.layer2, x)
        x = self.layer3(x)
        return x
```

**Trade-off:**
- **Memory**: Reduces by 30-50%
- **Speed**: Slower by 20-30% (recomputes activations)
- **Use when**: Model is too large but you can afford slower training

### 7. Use Efficient Data Loading

```python
# Efficient DataLoader settings
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,           # Parallel data loading
    pin_memory=True,         # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)

# Prefetch to GPU
for batch in train_loader:
    batch = batch.to('cuda', non_blocking=True)  # Async transfer
    # ... training code
```

### 8. Reduce Model Precision Strategically

```python
# Convert batch norm to FP32 (stability) while keeping model in FP16
model = model.half()  # FP16
for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.float()  # Keep BN in FP32
```

### 9. GPU-Agnostic Code Pattern

Write code that works on any GPU:

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Auto-detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Auto-enable AMP if GPU supports it
use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7

if use_amp:
    scaler = GradScaler()
    print("✅ Using mixed precision (FP16)")
else:
    scaler = None
    print("ℹ️  Using FP32")

# Training loop works with or without AMP
for batch in dataloader:
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass
    if use_amp:
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

### Quick Reference: Memory Optimization Checklist

- [ ] **Use mixed precision (FP16)** with `torch.cuda.amp` → 50% memory savings
- [ ] **Start with small batch size** (8-16) and find optimal with binary search
- [ ] **Use gradient accumulation** for effective large batch training
- [ ] **Enable gradient checkpointing** for very large models
- [ ] **Clear cache** between experiments: `torch.cuda.empty_cache()`
- [ ] **Delete unused tensors**: `del tensor; torch.cuda.empty_cache()`
- [ ] **Monitor memory** with `print_gpu_memory()` or `nvidia-smi`
- [ ] **Use `zero_grad(set_to_none=True)`** instead of `zero_grad()`
- [ ] **Efficient DataLoader**: `pin_memory=True`, `num_workers=4`
- [ ] **Save checkpoints** every N epochs to survive session timeouts
- [ ] **Test on CPU first**, then enable GPU to save quota

### Competition-Specific Considerations

Some Kaggle competitions have additional constraints:

- **Internet disabled** - Can't download external data/pretrained weights
- **Limited datasets** - Only competition-provided datasets allowed
- **Submission time limits** - Inference must complete in 2-9 hours
- **Code-only** - Notebooks run in restricted environment

Always check competition rules for specific hardware and time limits.

---

## Troubleshooting

### Authentication Failed

**Error:** `Failed to download or organize dataset`

**Solution:**
- **Local:** Ensure `~/.kaggle/kaggle.json` exists with correct permissions (600)
- **RunPod:** Set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables
- Verify credentials are correct from Kaggle website

### Dataset Not Found

**Error:** `Dataset not found`

**Solution:**
- Verify the dataset name in `config/train.yaml` matches Kaggle exactly
- Check dataset is public (not private)
- Ensure you've accepted any dataset terms on Kaggle website

### Symlink Already Exists

The script automatically handles existing symlinks/directories:
- If symlink exists: removes and recreates it
- If directory exists: skips symlink creation (shows warning)

To force recreation:
```bash
rm -rf kaggle_structure/data/train
python src/kaggle_structure/scripts/index.py
```

### Out of Memory (OOM) Errors

**Error:** `RuntimeError: CUDA out of memory`

**Solutions (in order):**
1. Reduce batch size: Try 16 → 8 → 4
2. Enable mixed precision (FP16) - saves 50% memory
3. Use gradient accumulation for effective larger batches
4. Enable gradient checkpointing
5. Clear cache: `torch.cuda.empty_cache()`
6. Use smaller model or fewer layers

### Session Timeout

**Error:** Notebook stops after 9-12 hours

**Solutions:**
- Save checkpoints every N epochs
- Resume training from checkpoints on restart
- Save to `/kaggle/working/` (persists for notebook lifetime)
- Break long training into multiple sessions

---

## Advanced Usage

### Using Different Config Files

```bash
# Modify resource_path in index.py to point to different configs
# config_path = resource_path("../config/production.yaml")
```

### Cleaning Up Cache

Kaggle datasets are cached in `~/.cache/kagglehub/`. To free space:

```bash
# Remove entire Kaggle cache
rm -rf ~/.cache/kagglehub/

# Remove specific dataset version
rm -rf ~/.cache/kagglehub/datasets/owner/dataset-name/
```

### Using Symlinks in Your Code

The symlinked data works exactly like regular directories:

```python
from pathlib import Path

data_dir = Path("kaggle_structure/data/train")

# List files
for file in data_dir.iterdir():
    print(file.name)

# Use with PyTorch
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.images = list(self.data_dir.glob("*.jpg"))
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        return image
    
    def __len__(self):
        return len(self.images)

dataset = MyDataset("kaggle_structure/data/train")
```

---

## License

This script uses the Kaggle API. Ensure you comply with:
- Kaggle's Terms of Service
- Individual dataset licenses
- Data usage restrictions

---

