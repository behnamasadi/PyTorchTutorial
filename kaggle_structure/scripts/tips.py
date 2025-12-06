import torch

# Always monitor memory usage


def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(
            f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


device = "cuda" if torch.cuda.is_available() else "cpu"


# Use this after model creation and during training

# Simple model
model = torch.nn.Linear(10, 2).to(device)
print_gpu_memory()


B, C, H, W = 16, 64, 224, 224
inputs = torch.randn(B, C, H, W)
inputs_test = inputs[:1].repeat(B, 1, 1, 1)
