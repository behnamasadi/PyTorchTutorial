import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple, List
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from PIL import Image
from utils.file_utils import resource_path
import torch
import torchvision.transforms as transforms
import matplotlib.pylab as plt
import pathlib
import os
import torch.nn as nn

# import torch.nn.modules
# import torch.utils.data.dataset

# Use the devkit's label list (NOT the module)


def pad_to_multiple(
    img_t: torch.Tensor,
    mask_t: torch.Tensor,
    multiple: int = 32,
    ignore_index: int = 255,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad image (reflect) & mask (constant=ignore) to next multiple of `multiple`."""
    h, w = img_t.shape[-2:]
    th = ((h + multiple - 1) // multiple) * multiple
    tw = ((w + multiple - 1) // multiple) * multiple
    ph, pw = th - h, tw - w
    if ph == 0 and pw == 0:
        return img_t, mask_t
    # pad=(left, right, top, bottom) -> push to right/bottom
    img_t = F.pad(img_t,  (0, pw, 0, ph), mode="reflect")
    mask_t = F.pad(mask_t, (0, pw, 0, ph), mode="constant", value=ignore_index)
    return img_t, mask_t


class TgsSaltSemanticDataset(Dataset):
    """
    Arguments:
      root: dataset root
      image_dir: relative dir with RGB PNGs
      label_dir: relative dir with label PNGs (values are raw ids)
      transform: image-only transform (e.g., normalization)
      pad_to: if not None (e.g., 32), pads image/mask to next multiple
    """

    def __init__(
        self,
        root: str,
        image_dir: str = "images",
        label_dir: str = "labels",
        transform: Optional[Callable] = None,
        pad_to: Optional[int] = 32
    ):
        self.root = Path(root)
        self.img_root = self.root.joinpath(image_dir)
        self.lbl_root = self.root.joinpath(label_dir)

        self.transform = transform
        self.pad_to = pad_to

        # Pair PNGs by stem
        self.samples: List[Tuple[Path, Path]] = []
        for img_path in sorted(self.img_root.rglob("*.png")):
            rel = img_path.relative_to(self.img_root)
            lbl_path = self.lbl_root.joinpath(rel)
            if lbl_path.exists():
                self.samples.append((img_path, lbl_path))
        if not self.samples:
            raise RuntimeError(
                f"No PNG pairs under {self.img_root} and {self.lbl_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):

        img_path, lbl_path = self.samples[idx]

        # --- Image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img_t = self.transform(img)     # user handles ToTensor + Normalize
        else:
            img_t = TF.to_tensor(img)       # just ToTensor

        # --- Mask (raw IDs -> trainIds)
        lbl = Image.open(lbl_path).convert("L")
        lbl_np = np.array(lbl, dtype=np.uint8)

        # Convert to binary segmentation: 0 -> 0 (background), non-zero -> 1 (foreground)
        # Keep 255 as ignore index
        lbl_train = np.where(lbl_np == 255, 255, np.where(lbl_np == 0, 0, 1))

        mask_t = torch.from_numpy(lbl_train).long()  # [H,W]

        padded_img_t, padded_mask_t = pad_to_multiple(
            img_t, mask_t, self.pad_to)

        return padded_img_t, padded_mask_t


################################################


if __name__ == "__main__":

    project_root = resource_path("")
    project_root = Path(project_root)
    print(project_root)

    data_path: str = "data/tgs_salt/trainSuper"

    root = project_root.joinpath(data_path)
    print(root)

    image_dir: str = "images"
    label_dir: str = "masks"

    img_root = root.joinpath(image_dir)
    lbl_root = root.joinpath(label_dir)

    pad_to = 32

    # # Typical image transform
    # img_tf = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.485, 0.456, 0.406),
    #                          std=(0.229, 0.224, 0.225)),
    # ])

    # Typical image transform
    img_tf = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = TgsSaltSemanticDataset(
        root, image_dir, label_dir, transform=img_tf, pad_to=pad_to)

    print(len(dataset))

    img, mask = dataset[0]
    print(img.shape)   # torch.Size([3,H,W]), float32
    print(mask.shape)  # torch.Size([H,W]), long with trainIds
    print(torch.unique(mask))  # unique class IDs, 0 or 255

    # Check a few more samples to see if we have class 1
    for i in range(min(5, len(dataset))):
        _, m = dataset[i]
        print(f"Sample {i} unique values: {torch.unique(m)}")
