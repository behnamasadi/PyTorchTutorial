import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader

# Detect optimal dtype
if torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
    print("dtype is bf16")
else:
    dtype = torch.float16
    print("dtype is f16")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Simple model
model = torch.nn.Linear(10, 2).to(device)

# Dummy dataset
X = torch.randn(40, 10)
Y = torch.randint(0, 2, (40,))     # CrossEntropyLoss expects class indices

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Use GradScaler only for FP16
use_scaler = (dtype == torch.float16)
scaler = GradScaler() if use_scaler else None

epochs = 10

for epoch in range(epochs):
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)

        with autocast(device_type='cuda', dtype=dtype):
            output = model(x)
            loss = criterion(output, y)

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    print(f"Epoch {epoch} finished")
