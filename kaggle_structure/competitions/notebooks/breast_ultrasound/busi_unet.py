"""Breast Ultrasound (BUSI) — tumor SEGMENTATION with U-Net + Dice (educational).

Segments breast lesions from ultrasound (benign / malignant; normal = no lesion).
U-Net (segmentation_models_pytorch), Dice+BCE, GT-vs-pred overlays. Runs on the 3090.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

CANDS = [
    "/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT",
    os.path.expanduser("~/.cache/kagglehub/datasets/aryashah2k/"
                       "breast-ultrasound-images-dataset/versions/1/Dataset_BUSI_with_GT"),
]
ROOT = Path(next(c for c in CANDS if Path(c).exists()))
OUT = Path("figs"); OUT.mkdir(exist_ok=True)


def safe_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        _ = nn.Conv2d(3, 4, 3).cuda()(torch.randn(1, 3, 8, 8, device="cuda"))
        torch.cuda.synchronize(); return torch.device("cuda")
    except Exception as e:
        print("GPU unusable, CPU:", str(e)[:70]); return torch.device("cpu")


DEV = safe_device(); print("device:", DEV)


def collect_pairs():
    pairs = []
    for img in ROOT.rglob("*.png"):
        if "_mask" in img.name:
            continue
        mask = img.with_name(img.stem + "_mask.png")
        if mask.exists():
            pairs.append((str(img), str(mask)))
    random.Random(42).shuffle(pairs)
    return pairs


class SegDS(Dataset):
    def __init__(self, pairs, size=128, train=False):
        self.pairs, self.size, self.train = pairs, size, train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = cv2.resize(cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB), (self.size, self.size))
        m = cv2.resize(cv2.imread(mp, cv2.IMREAD_GRAYSCALE), (self.size, self.size))
        if self.train and np.random.rand() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        y = torch.from_numpy(np.ascontiguousarray((m > 0).astype(np.float32)))[None]
        return (x - mean) / std, y


def dice_coef(pred, tgt, eps=1e-6):
    pred = (pred > 0.5).float(); inter = (pred * tgt).sum((1, 2, 3))
    return ((2 * inter + eps) / (pred.sum((1, 2, 3)) + tgt.sum((1, 2, 3)) + eps)).mean().item()


def eda(pairs):
    from collections import Counter
    cls = Counter(Path(p[0]).parent.name for p in pairs)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].bar(cls.keys(), cls.values(), color="#dd8452"); ax[0].set_title("BUSI classes")
    ip, mp = next(p for p in pairs if "malignant" in p[0])
    ax[1].imshow(cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)); ax[1].set_title("Ultrasound"); ax[1].axis("off")
    ax[2].imshow(cv2.imread(mp, 0), cmap="magma"); ax[2].set_title("Lesion mask"); ax[2].axis("off")
    fig.tight_layout(); fig.savefig(OUT / "eda.png", dpi=90); plt.close(fig)


def main():
    import segmentation_models_pytorch as smp
    pairs = collect_pairs()
    n_val = len(pairs) // 5
    va, tr = pairs[:n_val], pairs[n_val:]
    print(f"pairs={len(pairs)} train={len(tr)} val={len(va)}")
    eda(tr)
    dl_tr = DataLoader(SegDS(tr, train=True), 16, shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(SegDS(va), 16, num_workers=4, pin_memory=True)

    model = smp.Unet("efficientnet-b0", encoder_weights="imagenet", classes=1, activation=None).to(DEV)
    bce = nn.BCEWithLogitsLoss(); dice_loss = smp.losses.DiceLoss(mode="binary")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=DEV.type == "cuda")

    dices, EPOCHS = [], 12
    for ep in range(EPOCHS):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(DEV), y.to(DEV); opt.zero_grad()
            with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
                out = model(x); loss = bce(out, y) + dice_loss(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        model.eval(); ds = []
        with torch.inference_mode():
            for x, y in dl_va:
                ds.append(dice_coef(torch.sigmoid(model(x.to(DEV))).cpu(), y))
        dices.append(float(np.mean(ds))); print(f"epoch {ep+1}/{EPOCHS} val_dice={dices[-1]:.4f}", flush=True)

    print(f"\nBEST val Dice = {max(dices):.4f}")
    plt.figure(figsize=(6, 4)); plt.plot(range(1, EPOCHS+1), dices, "o-", color="#dd8452")
    plt.title("Validation Dice"); plt.xlabel("epoch"); plt.savefig(OUT / "dice.png", dpi=90); plt.close()

    overlays(model, [p for p in va if cv2.imread(p[1], 0).max() > 0])
    torch.save(model.state_dict(), OUT / "model.pt")
    print("saved all figures to", OUT)


def overlays(model, pairs, n=4):
    ds = SegDS(pairs); model.eval()
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for r, idx in enumerate(np.linspace(0, len(ds)-1, n).astype(int)):
        x, y = ds[idx]
        with torch.inference_mode():
            pred = torch.sigmoid(model(x[None].to(DEV)))[0, 0].cpu().numpy()
        rgb = np.clip(x.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) +
                      np.array([0.485, 0.456, 0.406]), 0, 1)
        axes[r, 0].imshow(rgb); axes[r, 0].set_title("Ultrasound"); axes[r, 0].axis("off")
        axes[r, 1].imshow(rgb); axes[r, 1].imshow(y[0], cmap="Reds", alpha=0.5)
        axes[r, 1].set_title("Ground truth"); axes[r, 1].axis("off")
        axes[r, 2].imshow(rgb); axes[r, 2].imshow(pred > 0.5, cmap="Greens", alpha=0.5)
        axes[r, 2].set_title("Prediction"); axes[r, 2].axis("off")
    fig.suptitle("U-Net breast-lesion segmentation — GT vs prediction")
    fig.tight_layout(); fig.savefig(OUT / "overlays.png", dpi=90); plt.close(fig)
    print("saved overlays")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high"); main()
