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

