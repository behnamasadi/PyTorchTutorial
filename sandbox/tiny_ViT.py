import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.datasets.cifar

# torchvision.models.vgg16(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Tiny Vision Transformer Configuration
image_size = 32        # 32x32 image
patch_size = 8         # 8x8 patches
num_patches = (image_size // patch_size) ** 2
embedding_dim = 64     # dimension of patch embedding
num_heads = 4
num_layers = 2
mlp_dim = 128
num_classes = 10       # for example, CIFAR-10


class PatchEmbedding(torch.nn.Module):
    def __init__(self, in_channels=3, patch_size=8, emb_dim=64):
        super().__init__()

        self.proj = nn.Conv2d(in_channels, emb_dim,
                              kernel_size=patch_size, stride=patch_size)

        self.fc1 = nn.Linear(in_features=4, out_features=4)
        pass

    def forward(self, x):
        x = self.proj(x)  # [B, emb_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, emb_dim]
        return x


# Input image: shape (1, 1, 3, 3)
img = torch.tensor([[[[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]]]], dtype=torch.float32)

print("Image shape:", img.shape)  # (N=1, C=1, H=3, W=3)

# Kernel: shape (2, 2), no channel yet
kernel = torch.tensor([[1, 0],
                       [0, -1]], dtype=torch.float32)


unfold = torch.nn.Unfold(kernel_size=(2, 2))  # no padding, stride=1
patches = unfold(img)  # shape: (N, C*kH*kW, L), here (1, 4, 4)
print("Unfolded patches:\n", patches)


# einops.rearrange
# https://einops.rocks/api/rearrange/#:~:text=rearrange-,einops.,stack%2C%20concatenate%20and%20other%20operations.
