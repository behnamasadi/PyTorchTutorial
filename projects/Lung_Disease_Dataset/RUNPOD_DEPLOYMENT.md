# RunPod Deployment Guide

This guide explains how to deploy and train the Lung Disease Dataset project on RunPod.

## Prerequisites

1. **RunPod Account**: Sign up at [runpod.io](https://www.runpod.io)
2. **Docker Hub Account** (optional): For pushing your Docker image
3. **Dataset**: The Lung Disease Dataset should be available (can be downloaded via Kaggle)

## Step 1: Prepare Your Project

### Option A: Using Git Repository (Recommended)

1. Push your project to GitHub/GitLab
2. RunPod can clone directly from your repository

### Option B: Using Docker Image

1. Build and push your Docker image to Docker Hub:

```bash
# Build the Docker image
docker build -t your-username/lung-disease-training:latest .

# Push to Docker Hub
docker push your-username/lung-disease-training:latest
```

## Step 2: Create RunPod Template

1. Go to RunPod Dashboard → **Templates**
2. Click **New Template**
3. Configure the template:

### Template Configuration

**Template Name**: `Lung Disease Training`

**Container Image**: 
- If using Docker Hub: `your-username/lung-disease-training:latest`
- If using Git: Leave empty (will use base image)

**Container Disk**: `20 GB` (minimum, increase if needed)

**Docker Command**:
```bash
bash runpod_start.sh
```

**Environment Variables** (optional):
```
CONFIG_FILE=configs/train.yaml
DEVICE=cuda:0
```

**Volume Mounts** (if using persistent storage):
- `/workspace/data` → Your dataset path
- `/workspace/checkpoints` → For saving models
- `/workspace/runs` → For TensorBoard logs

## Step 3: Configure RunPod Pod

1. Go to **Pods** → **Deploy Pod**
2. Select your template
3. Choose GPU:
   - **RTX 3090** (24GB) - Good for most models
   - **RTX 4090** (24GB) - Faster training
   - **A100** (40GB/80GB) - For larger models
4. Set **Container Disk**: `20-50 GB` depending on your needs
5. Configure **Network Volume** (optional but recommended):
   - Mount point: `/workspace/data`
   - Size: Based on your dataset size

## Step 4: Prepare Dataset

### Option A: Upload to Network Volume

1. After pod is created, connect via SSH or Jupyter
2. Upload your dataset to the mounted volume:
   ```bash
   # Dataset structure should be:
   /workspace/data/
     ├── train/
     │   ├── Bacterial Pneumonia/
     │   ├── Corona Virus Disease/
     │   ├── Normal/
     │   ├── Tuberculosis/
     │   └── Viral Pneumonia/
     ├── val/
     │   └── (same structure)
     └── test/
         └── (same structure)
   ```

### Option B: Download via Kaggle

1. Set up Kaggle API credentials:
   ```bash
   # In RunPod terminal
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   ```

2. Download dataset:
   ```python
   python -c "import kagglehub; path = kagglehub.dataset_download('omkarmanohardalvi/lungs-disease-dataset-4-types'); print(f'Downloaded to: {path}')"
   ```

## Step 5: Start Training

### Via RunPod UI

1. Deploy your pod with the template
2. The training will start automatically when the pod starts

### Via SSH/Terminal

1. Connect to your pod via SSH
2. Navigate to project directory:
   ```bash
   cd /workspace
   ```

3. Run training manually:
   ```bash
   python scripts/train.py --config configs/train.yaml --device cuda:0
   ```

### Custom Configuration

You can override the default config:

```bash
python scripts/train.py \
    --config configs/train.yaml \
    --device cuda:0
```

Or set environment variables before starting:
```bash
export CONFIG_FILE=configs/train.yaml
export DEVICE=cuda:0
bash runpod_start.sh
```

## Step 6: Monitor Training

### TensorBoard

1. Start TensorBoard in the pod:
   ```bash
   tensorboard --logdir=runs --host=0.0.0.0 --port=6006
   ```

2. Access via RunPod's port forwarding or public URL

### Weights & Biases

1. Login to W&B (if not already):
   ```bash
   wandb login
   ```

2. Training will automatically log to W&B if configured in `configs/train.yaml`

### Check Logs

Training output is printed to stdout. You can also check:
- `checkpoints/` - Model checkpoints
- `runs/` - TensorBoard logs
- `wandb/` - W&B logs (if enabled)

## Step 7: Save Results

### Download Checkpoints

1. Use RunPod's file browser to download checkpoints
2. Or use SCP/SFTP to transfer files:
   ```bash
   scp -r user@pod-ip:/workspace/checkpoints ./local_checkpoints
   ```

### Network Volume

If you mounted a network volume, checkpoints are automatically saved there and persist after pod termination.

## Configuration Tips

### Adjust Batch Size for GPU Memory

Edit `configs/model.yaml` to adjust batch sizes based on your GPU:

```yaml
models:
  convnextv2_tiny:
    batch_size: 32  # Increase for larger GPUs
  tf_efficientnetv2_m:
    batch_size: 16  # Adjust based on GPU memory
```

### Multi-GPU Training

For multi-GPU setups, you may need to modify the training script to use `torch.nn.DataParallel` or `torch.distributed`.

### Memory Optimization

If you encounter OOM errors:
1. Reduce batch size in `configs/model.yaml`
2. Enable gradient checkpointing (if implemented)
3. Use mixed precision training (if stable)

## Troubleshooting

### Container Fails to Start

- Check Docker logs in RunPod dashboard
- Verify `runpod_start.sh` has execute permissions
- Ensure all dependencies are in `requirements.txt`

### CUDA Out of Memory

- Reduce batch size in model config
- Use a smaller model
- Enable gradient accumulation

### Dataset Not Found

- Verify dataset path in `configs/data.yaml`
- Check volume mounts are correct
- Ensure dataset is uploaded/downloaded

### Training Too Slow

- Use a faster GPU (RTX 4090, A100)
- Increase batch size if memory allows
- Reduce image size in config
- Use fewer data augmentation

## Cost Optimization

1. **Use Spot Instances**: Cheaper but can be interrupted
2. **Stop Pod When Idle**: RunPod charges per hour
3. **Use Appropriate GPU**: Don't over-provision
4. **Monitor Training**: Stop early if not improving

## Example RunPod Template JSON

```json
{
  "name": "Lung Disease Training",
  "imageName": "your-username/lung-disease-training:latest",
  "dockerArgs": "",
  "containerDiskInGb": 20,
  "volumeInGb": 0,
  "volumeMountPath": "/workspace/data",
  "env": [
    {
      "key": "CONFIG_FILE",
      "value": "configs/train.yaml"
    },
    {
      "key": "DEVICE",
      "value": "cuda:0"
    }
  ],
  "ports": "6006/tcp",
  "startJupyter": false,
  "startSsh": true
}
```

## Support

For issues specific to:
- **RunPod**: Check [RunPod Documentation](https://docs.runpod.io)
- **Project**: Check project README.md
- **Training**: Review training logs and configs

