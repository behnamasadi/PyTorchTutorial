import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.datasets.cifar


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
