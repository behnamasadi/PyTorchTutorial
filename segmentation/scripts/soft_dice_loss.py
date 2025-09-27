import torch
import torch.nn as nn

# C is the number of classes and NOT the number of image inputs, so 1 batch, 3 channels (for class A, B, C) of 2x2 tensors
N, C, H, W = 1, 3, 2, 2
logits = torch.zeros(N, C, H, W)


# our logits are [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0]
# This means for the first pixel at [0,0], our model output is 1 for class A, 0 for class B, and 0 for class C
# Fill logits per pixel (A,B,C) = channels 0,1,2
logits[0, :, 0, 0] = torch.tensor([1.0, 0.0, 0.0])   # (0,0)
logits[0, :, 0, 1] = torch.tensor([0.0, 1.0, 0.0])   # (0,1)
logits[0, :, 1, 0] = torch.tensor([0.0, 0.0, 1.0])   # (1,0)
logits[0, :, 1, 1] = torch.tensor([0.5, 0.5, 0.0])   # (1,1)

# CrossEntropyLoss expects class IDs in [0, C−1].
# A = 0
# B = 1
# C = 2

# Target shape must include the batch dim
gt = torch.zeros((1, 2, 2), dtype=torch.long)
gt[0, 0, 0] = 0
gt[0, 0, 1] = 1
gt[0, 1, 0] = 1
gt[0, 1, 1] = 2


# dim=1 mean across channels and NOT across H or W
# softmax per pixel
probs = torch.softmax(logits, dim=1)
print("probs(0,0):", probs[0, :, 0, 0])
print("probs(0,1):", probs[0, :, 0, 1])
print("probs(1,0):", probs[0, :, 1, 0])
print("probs(1,1):", probs[0, :, 1, 1])


# targets: (N, H, W) int64 in [0..C-1], optional ignore_index
targets = gt
ignore_index = 255
valid = (targets != ignore_index)  # (N,H,W)
print("valid:\n", valid)


probs = probs * valid.unsqueeze(1)
print("probs.shape:\n", probs.shape)

print("probs:\n", probs)

# one-hot targets on valid pixels
one_hot = torch.zeros_like(probs).scatter_(
    1, targets.clamp(0, probs.size(1)-1).unsqueeze(1), 1.0)
one_hot = one_hot * valid.unsqueeze(1)  # zero out ignored pixels

print("one_hot\n", one_hot)
print(one_hot[0, 0, :, :])
print(one_hot[0, 1, :, :])
print(one_hot[0, 2, :, :])


eps = 1e-6
p_sum = probs.sum(dim=(0, 2, 3))             # per-class sum of probabilities
print("p_sum:\n", p_sum)
y_sum = one_hot.sum(dim=(0, 2, 3))           # per-class sum of GT pixels
inter = (probs * one_hot).sum(dim=(0, 2, 3))  # per-class soft intersection

dice_per_class = (2*inter + eps) / (p_sum + y_sum + eps)
dice_macro = dice_per_class.mean()
soft_dice_loss = 1 - dice_macro

print("soft dice loss:", soft_dice_loss)
