import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets, ignore_index=255):
        # logits: (N, C, H, W)
        # targets: (N, H, W), int64 class indices

        probs = F.softmax(logits, dim=1)

        valid = (targets != ignore_index)
        one_hot = torch.zeros_like(probs).scatter_(
            1, targets.clamp(0, probs.size(1)-1).unsqueeze(1), 1.0
        )
        one_hot = one_hot * valid.unsqueeze(1)
        probs = probs * valid.unsqueeze(1)

        p_sum = probs.sum(dim=(0, 2, 3))
        y_sum = one_hot.sum(dim=(0, 2, 3))
        inter = (probs * one_hot).sum(dim=(0, 2, 3))

        dice_per_class = (2*inter + self.eps) / (p_sum + y_sum + self.eps)
        return 1 - dice_per_class.mean()


if __name__ == "__main__":
    # Example usage
    criterion = SoftDiceLoss()
    ce = nn.CrossEntropyLoss(ignore_index=255)

    logits = torch.randn(2, 3, 128, 128)  # batch=2, classes=3
    targets = torch.randint(0, 3, (2, 128, 128))

    loss = ce(logits, targets) + criterion(logits, targets)
