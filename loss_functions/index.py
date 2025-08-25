import torch
import torch.nn as nn

correct = 0
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction="mean")
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction="sum")
criterion = nn.CrossEntropyLoss()

# Batch of size 2, 3 classes → logits
outputs = torch.tensor([[0.1, 0.3, 0.2], [0.4, 0.1, 0.2]])  # shape (2,3)

# True class index (not one-hot!)
label = torch.tensor([2, 0])  # shape (2,)

loss = criterion(outputs, label)
print("outputs.shape", outputs.shape)
print("outputs", outputs)
_, predicted = outputs.max(1)
# _, predicted = torch.max(outputs, 1)
print("predicted", predicted)

correct += predicted.eq(label).sum().item()
print(predicted.eq(label).sum().item())
print(predicted)
print(correct)
print("-"*50)
print(torch.max(outputs, dim=1))


# print(loss.item())


# preds = model(x)
# loss = criterion(preds, y)
