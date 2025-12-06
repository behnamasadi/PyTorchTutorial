import subprocess
import torch
import timm
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler

# Clear any existing GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_gpu_memory_from_nvidia_smi():
    """
    Returns total and used VRAM from nvidia-smi in MB.
    """
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total',
         '--format=csv,nounits,noheader']
    )
    used, total = map(int, result.decode().strip().split(','))
    return used, total


def monitor(step=""):
    used, total = get_gpu_memory_from_nvidia_smi()

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2

    print(f"\n=== {step} ===")
    print(f"nvidia-smi used      : {used:.1f} MB / {total:.1f} MB")
    print(f"PyTorch allocated    : {allocated:.1f} MB")
    print(f"PyTorch reserved     : {reserved:.1f} MB")
    print(f"PyTorch max allocated: {max_alloc:.1f} MB")


device = "cuda" if torch.cuda.is_available() else "cpu"


model_name = "tf_efficientnetv2_s"
model = timm.create_model(model_name=model_name, pretrained=True).to(device)

# Get model configuration
cfg = model.default_cfg


C, H, W = list(cfg['input_size'])

num_class = cfg['num_classes']
num_samples = 100
X = torch.randn(num_samples, C, H, W)
Y = torch.randint(0, num_class, (num_samples,))

dataset = TensorDataset(X, Y)
batch_size = 10  # Reduced from 20 to leave room for gradient accumulation

data_loader = DataLoader(batch_size=batch_size, dataset=dataset,
                         pin_memory=True, num_workers=4)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


# Detect optimal dtype
if torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
    print("dtype is bf16")
else:
    dtype = torch.float16
    print("dtype is f16")


# Use GradScaler only for FP16
use_scaler = (dtype == torch.float16)
scaler = GradScaler() if use_scaler else None


accum_steps = 8   # Effective batch size = 10 * 8 = 80 (simulated larger batch)
model.train()

optimizer.zero_grad()

for step, (images, labels) in enumerate(data_loader):
    # Move data to device
    images = images.to(device)
    labels = labels.to(device)

    # Use autocast with detected dtype (bf16 or fp16)
    with torch.amp.autocast(device == device, dtype=dtype):
        outputs = model(images)
        loss = criterion(outputs, labels)

    loss = loss / accum_steps   # scale the loss

    # Use GradScaler for FP16, regular backward for BF16
    if use_scaler:
        scaler.scale(loss).backward()
    else:
        loss.backward()             # gradients accumulate

    # Clean up to save memory
    del outputs, loss, images, labels

    if (step + 1) % accum_steps == 0:
        if use_scaler:
            scaler.step(optimizer)  # update weights
            scaler.update()
            optimizer.zero_grad()   # reset for next accumulation
        else:
            optimizer.step()        # update weights
            optimizer.zero_grad()   # reset for next accumulation

        # Optional: clear cache after each optimization step
        torch.cuda.empty_cache()
