"""LGG Brain MRI — tumor SEGMENTATION with U-Net + Dice (educational).

Segments low-grade-glioma tumors from FLAIR brain MRI. U-Net (segmentation_models_
pytorch, EfficientNet encoder), Dice+BCE loss, patient-level split (no leakage).
Visualizes image | ground-truth | prediction overlays — the segmentation analog of Grad-CAM.
"""
from __future__ import annotations

import os
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
    "/kaggle/input/lgg-mri-segmentation/kaggle_3m",
    os.path.expanduser("~/.cache/kagglehub/datasets/mateuszbuda/"
                       "lgg-mri-segmentation/versions/2/kaggle_3m"),
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
    """(image, mask) pairs grouped by patient for a leak-free split."""
    by_patient = {}
    for mask in ROOT.rglob("*_mask.tif"):
        img = Path(str(mask).replace("_mask", ""))
        if img.exists():
            by_patient.setdefault(mask.parent.name, []).append((str(img), str(mask)))
    return by_patient


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
    pred = (pred > 0.5).float()
    inter = (pred * tgt).sum((1, 2, 3))
    return ((2 * inter + eps) / (pred.sum((1, 2, 3)) + tgt.sum((1, 2, 3)) + eps)).mean().item()


def eda(pairs):
    tumor = [p for p in pairs if cv2.imread(p[1], 0).max() > 0]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].bar(["with tumor", "no tumor"], [len(tumor), len(pairs)-len(tumor)], color="#55a868")
    ax[0].set_title("Slices with/without tumor")
    ip, mp = tumor[len(tumor)//2]
    ax[1].imshow(cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)); ax[1].set_title("FLAIR MRI"); ax[1].axis("off")
    ax[2].imshow(cv2.imread(mp, 0), cmap="magma"); ax[2].set_title("Tumor mask (label)"); ax[2].axis("off")
    fig.tight_layout(); fig.savefig(OUT / "eda.png", dpi=90); plt.close(fig)


def main():
    import segmentation_models_pytorch as smp
    by_patient = collect_pairs()
    patients = sorted(by_patient)
    n_val = max(1, len(patients) // 5)
    val_p, tr_p = patients[:n_val], patients[n_val:]
    tr = [pr for p in tr_p for pr in by_patient[p]]
    va = [pr for p in val_p for pr in by_patient[p]]
    print(f"patients={len(patients)} train_slices={len(tr)} val_slices={len(va)}")
    eda(tr)

    dl_tr = DataLoader(SegDS(tr, train=True), 32, shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(SegDS(va), 32, num_workers=4, pin_memory=True)

    model = smp.Unet("efficientnet-b0", encoder_weights="imagenet", classes=1, activation=None).to(DEV)
    bce = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode="binary")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=DEV.type == "cuda")

    hist, dices, EPOCHS = [], [], 8
    for ep in range(EPOCHS):
        model.train(); tot = 0
        for x, y in dl_tr:
            x, y = x.to(DEV), y.to(DEV); opt.zero_grad()
            with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
                out = model(x); loss = bce(out, y) + dice_loss(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item()
        model.eval(); ds = []
        with torch.inference_mode():
            for x, y in dl_va:
                ds.append(dice_coef(torch.sigmoid(model(x.to(DEV))).cpu(), y))
        hist.append(tot/len(dl_tr)); dices.append(float(np.mean(ds)))
        print(f"epoch {ep+1}/{EPOCHS} loss={hist[-1]:.4f} val_dice={dices[-1]:.4f}", flush=True)

    print(f"\nBEST val Dice = {max(dices):.4f}")
    plt.figure(figsize=(6, 4)); plt.plot(range(1, EPOCHS+1), dices, "o-", color="#55a868")
    plt.title("Validation Dice"); plt.xlabel("epoch"); plt.savefig(OUT / "dice.png", dpi=90); plt.close()

    overlays(model, [p for p in va if cv2.imread(p[1], 0).max() > 0])
    torch.save(model.state_dict(), OUT / "model.pt")
    print("saved all figures to", OUT)


def overlays(model, pairs, n=4):
    """image | ground-truth | prediction — the segmentation explainability figure."""
    ds = SegDS(pairs); model.eval()
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for r, idx in enumerate(np.linspace(0, len(ds)-1, n).astype(int)):
        x, y = ds[idx]
        with torch.inference_mode():
            pred = torch.sigmoid(model(x[None].to(DEV)))[0, 0].cpu().numpy()
        rgb = np.clip(x.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.206]) +
                      np.array([0.485, 0.456, 0.406]), 0, 1)
        axes[r, 0].imshow(rgb); axes[r, 0].set_title("MRI"); axes[r, 0].axis("off")
        axes[r, 1].imshow(rgb); axes[r, 1].imshow(y[0], cmap="Reds", alpha=0.5)
        axes[r, 1].set_title("Ground truth"); axes[r, 1].axis("off")
        axes[r, 2].imshow(rgb); axes[r, 2].imshow(pred > 0.5, cmap="Greens", alpha=0.5)
        axes[r, 2].set_title("Prediction"); axes[r, 2].axis("off")
    fig.suptitle("U-Net tumor segmentation — GT vs prediction")
    fig.tight_layout(); fig.savefig(OUT / "overlays.png", dpi=90); plt.close(fig)
    print("saved overlays")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high"); main()
