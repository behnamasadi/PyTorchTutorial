import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Resize,
    ScaleIntensity, ToTensor
)

from monai.data import Dataset
from monai.networks.nets import DenseNet121  # a MONAI model (PyTorch-based)


# ---------------------------------------------------------
# 1. MONAI TRANSFORMS
# ---------------------------------------------------------
train_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Resize((224, 224)),
    ScaleIntensity(),
    ToTensor(),
])

val_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Resize((224, 224)),
    ScaleIntensity(),
    ToTensor(),
])

# ---------------------------------------------------------
# 2. DATASET
# ---------------------------------------------------------
train_files = [
    {"image": "/path/to/img1.png", "label": 0},
    {"image": "/path/to/img2.png", "label": 1},
]

val_files = [
    {"image": "/path/to/img3.png", "label": 0},
    {"image": "/path/to/img4.png", "label": 1},
]


train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)


model = DenseNet121(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
)

model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

batch_size = 1

for id, train_item in enumerate(train_files):
    for batch_ix in range(batch_size):
        print(train_item)
