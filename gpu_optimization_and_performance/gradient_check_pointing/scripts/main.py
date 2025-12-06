import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import time
import subprocess


def get_gpu_memory_from_nvidia_smi():
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


# --------------------------------------
# Models
# --------------------------------------
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1024, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)

        x = self.layer2(x)
        x = self.act(x)

        x = self.layer3(x)
        return x


class MyModelCheckpointed(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1024, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 10)
        self.act = nn.ReLU()

    def forward_block1(self, x):
        x = self.layer1(x)
        x = self.act(x)
        return x

    def forward_block2(self, x):
        x = self.layer2(x)
        x = self.act(x)
        return x

    def forward(self, x):
        x = checkpoint.checkpoint(self.forward_block1, x)
        x = checkpoint.checkpoint(self.forward_block2, x)
        x = self.layer3(x)
        return x


monitor("start")

B = 2048
x = torch.rand(B, 1024).to(device)
y = torch.randint(0, 10, (B,)).to(device)

criterion = nn.CrossEntropyLoss()


# --------------------------------------
# Without checkpointing
# --------------------------------------
model_no_checkpointing = MyModel().to(device)
optimizer = torch.optim.Adam(model_no_checkpointing.parameters(), lr=1e-3)

torch.cuda.reset_peak_memory_stats()
monitor("before no-checkpoint train step")

optimizer.zero_grad()
out = model_no_checkpointing(x)
loss = criterion(out, y)
loss.backward()
optimizer.step()

monitor("after no-checkpoint train step")


# --------------------------------------
# With checkpointing
# --------------------------------------
model_checkpointing = MyModelCheckpointed().to(device)
optimizer = torch.optim.Adam(model_checkpointing.parameters(), lr=1e-3)

torch.cuda.reset_peak_memory_stats()
monitor("before checkpointed train step")

optimizer.zero_grad()
out = model_checkpointing(x)
loss = criterion(out, y)
loss.backward()
optimizer.step()

monitor("after checkpointed train step")
