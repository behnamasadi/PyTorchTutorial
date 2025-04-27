import torch
import torch.nn as nn
import torch.optim as optim
import torch.scheduler as scheduler

model = nn.Linear(1, 1)
optim = optim.SGD(model.parameters(), lr=0.01)
scheduler = scheduler.StepLR(optim, step_size=100, gamma=0.1)
scheduler = scheduler.MultiStepLR(optim, milestones=[100, 200], gamma=0.1)

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    scheduler.step()
