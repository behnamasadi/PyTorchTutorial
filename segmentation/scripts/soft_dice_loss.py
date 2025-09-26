import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # y_pred: (N, 1, H, W) after sigmoid
        # y_true: (N, 1, H, W) with values 0 or 1
        y_pred = y_pred.contiguous().view(y_pred.size(0), -1)
        y_true = y_true.contiguous().view(y_true.size(0), -1)

        intersection = (y_pred * y_true).sum(dim=1)
        dice_score = (2. * intersection + self.epsilon) / \
                     (y_pred.sum(dim=1) + y_true.sum(dim=1) + self.epsilon)
        loss = 1 - dice_score
        return loss.mean()


N, C, H, W = 1, 1, 2, 2
logits = torch.zeros([N, C, H, W])

print(logits)
print(logits.shape)

exit()

logits = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0]])

output_softmax = torch.softmax(logits, dim=1)
print(output_softmax)

A = 1
B = 2
C = 3
gt = torch.tensor([A, B, B, C])
# print(torch.unique(gt))
print(gt)

# #----

# # logits: (N, C, H, W)
# probs = torch.softmax(logits, dim=1)  # turn logits -> probabilities

# # targets: (N, H, W) int64 in [0..C-1], optional ignore_index
# ignore_index = 255
# valid = (targets != ignore_index)  # (N,H,W)

# # one-hot targets on valid pixels
# one_hot = torch.zeros_like(probs).scatter_(1, targets.clamp(0, probs.size(1)-1).unsqueeze(1), 1.0)
# one_hot = one_hot * valid.unsqueeze(1)  # zero out ignored pixels
# probs   = probs   * valid.unsqueeze(1)

# eps = 1e-6
# p_sum = probs.sum(dim=(0,2,3))             # per-class sum of probabilities
# y_sum = one_hot.sum(dim=(0,2,3))           # per-class sum of GT pixels
# inter = (probs * one_hot).sum(dim=(0,2,3)) # per-class soft intersection

# dice_per_class = (2*inter + eps) / (p_sum + y_sum + eps)
# dice_macro = dice_per_class.mean()
# soft_dice_loss = 1 - dice_macro
