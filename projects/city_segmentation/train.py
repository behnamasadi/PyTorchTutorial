from pathlib import Path
from typing import Callable, Optional, Tuple, List
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

# From KITTI devkit (you already have these):
# utils/devkit/helpers.py
from utils.devkit.helpers.labels import labels as LABELS

# from utils.devkit.helpers.labels import labels, name2label, id2label, trainId2label


def _build_id_to_trainid_lut(ignore_index: int = 255) -> np.ndarray:
    """Vectorized id->trainId mapping using the KITTI/Cityscapes devkit labels."""
    lut = np.full(256, ignore_index, dtype=np.uint8)
    for lab in LABELS:
        if 0 <= lab.id <= 255:
            lut[lab.id] = lab.trainId
    return lut


def _pad_to_multiple(
    img_t: torch.Tensor, mask_t: torch.Tensor, multiple: int = 32, ignore_index: int = 255
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad image (reflect) and mask (constant=ignore_index) to next multiple of `multiple`."""
    h, w = img_t.shape[-2:]
    th = ((h + multiple - 1) // multiple) * multiple
    tw = ((w + multiple - 1) // multiple) * multiple
    ph, pw = th - h, tw - w
    if ph == 0 and pw == 0:
        return img_t, mask_t
    # pad = (left, right, top, bottom) — push to right/bottom
    img_t = F.pad(img_t,  (0, pw, 0, ph), mode="reflect")
    mask_t = F.pad(mask_t, (0, pw, 0, ph), mode="constant", value=ignore_index)
    return img_t, mask_t


class KittiSemanticDataset(Dataset):
    """
    Minimal KITTI semantic dataset with id->trainId remap and optional padding.
    Directory layout is configurable; filenames must match by stem.

    Args:
        root: dataset root
        image_dir: relative path for RGB images (e.g., 'training/image_2')
        label_dir: relative path for label PNGs with 'id' values (e.g., 'training/semantic')
        transform: image-only transform (applied after joint_transform). Input is PIL.Image or tensor depending on your pipeline.
        target_transform: mask-only transform (applied after joint_transform). Input is a torch.LongTensor mask.
        joint_transform: callable(img_pil, mask_pil) -> (img_pil, mask_pil) for paired ops (resize/crop/flip).
        to_tensor_and_normalize: if True, converts image to tensor [0,1] and normalizes with given mean/std.
        mean/std: normalization stats (ImageNet by default).
        pad_to_multiple: if set (e.g., 32), pads image (reflect) & mask (255) to that multiple.
        ignore_index: index used for ignored pixels (default 255).
    """

    def __init__(
        self,
        root: str,
        image_dir: str = "training/image_2",
        label_dir: str = "training/semantic",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        joint_transform: Optional[Callable] = None,
        to_tensor_and_normalize: bool = True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        pad_to_multiple: Optional[int] = 32,
        ignore_index: int = 255,
    ):
        self.root = Path(root)
        self.img_root = self.root / image_dir
        self.lbl_root = self.root / label_dir

        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.to_tensor_and_normalize = to_tensor_and_normalize
        self.mean = mean
        self.std = std
        self.pad_to_multiple = pad_to_multiple
        self.ignore_index = ignore_index

        # Pair images and labels by identical relative stem
        img_exts = {".png", ".jpg", ".jpeg"}
        self.samples: List[Tuple[Path, Path]] = []
        for img_path in sorted(self.img_root.rglob("*")):
            if img_path.suffix.lower() in img_exts:
                rel = img_path.relative_to(self.img_root)
                lbl_path = (self.lbl_root / rel).with_suffix(".png")
                if lbl_path.exists():
                    self.samples.append((img_path, lbl_path))
        if not self.samples:
            raise RuntimeError(
                f"No (image,label) pairs under {self.img_root} and {self.lbl_root}")

        # Build LUT once
        self.id2train = _build_id_to_trainid_lut(
            ignore_index=self.ignore_index)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]

        # Load PIL
        img_pil = Image.open(img_path).convert("RGB")
        lbl_pil = Image.open(lbl_path).convert("L")  # id mask

        # Joint transforms (keep them deterministic & paired)
        if self.joint_transform is not None:
            img_pil, lbl_pil = self.joint_transform(img_pil, lbl_pil)

        # Convert to tensors early so we can do safe LUT, padding, normalization
        img_t = TF.to_tensor(img_pil) if self.to_tensor_and_normalize else TF.pil_to_tensor(
            img_pil).float().div(255)
        # id->trainId remap (vectorized)
        lbl_np = np.array(lbl_pil, dtype=np.uint8)
        lbl_train = self.id2train[lbl_np]              # (H,W) uint8
        mask_t = torch.from_numpy(lbl_train).long()    # [H,W] long

        # Optional normalization
        if self.to_tensor_and_normalize:
            img_t = TF.normalize(img_t, self.mean, self.std)

        # Optional padding to multiple
        if self.pad_to_multiple is not None:
            img_t, mask_t = _pad_to_multiple(
                img_t, mask_t, multiple=self.pad_to_multiple, ignore_index=self.ignore_index)

        # Image-only and mask-only transforms (post-joint)
        if self.transform is not None:
            img_t = self.transform(img_t)
        if self.target_transform is not None:
            mask_t = self.target_transform(mask_t)

        return img_t, mask_t, str(img_path)


if __name__ == "__main__":
    lut = _build_id_to_trainid_lut()

    print("trainId for car (id=26):", lut[26])  # e.g., 13
