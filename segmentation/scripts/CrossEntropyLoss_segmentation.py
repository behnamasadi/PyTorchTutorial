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


criterion = nn.CrossEntropyLoss(ignore_index=255)

loss = criterion(logits, gt)
print("CrossEntropyLoss:", loss.item())
