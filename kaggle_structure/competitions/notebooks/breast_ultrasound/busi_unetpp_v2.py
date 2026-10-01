"""Breast Ultrasound (BUSI) — IMPROVED segmentation (v2).

Upgrades over v1 (plain U-Net / eff-b0 / 128px / hflip / 12ep -> Dice 0.69):
  - U-Net++ decoder, EfficientNet-b4 encoder (imagenet)
  - 256px (was 128)
  - real augmentation (albumentations): flips, shift-scale-rotate, brightness/contrast, gauss noise
  - Dice + BCE loss, AdamW + cosine LR, 60 epochs
  - lesion-bearing images only (benign+malignant) = the standard BUSI seg benchmark
    (normal images have empty masks and artificially inflate Dice)
  - multiple masks per image merged (OR)
Fair comparison: also reports v1-style plain-U-Net baseline Dice on the SAME lesion-only split.
"""
from __future__ import annotations
import os, random
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import albumentations as A
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset

ROOT = Path(os.path.expanduser("~/.cache/kagglehub/datasets/aryashah2k/"
            "breast-ultrasound-images-dataset/versions/1/Dataset_BUSI_with_GT"))
OUT = Path(__file__).parent / "figs_v2"; OUT.mkdir(exist_ok=True)
DEV = torch.device("cuda"); SIZE = 256
print("device:", DEV, "| torch", torch.__version__)


def collect_pairs(lesion_only=True):
    """Return (image, [mask,...]) for each base image. Merge multi-mask cases."""
    pairs = []
    for img in ROOT.rglob("*.png"):
        if "_mask" in img.name:
            continue
        cls = img.parent.name  # benign / malignant / normal
        if lesion_only and cls == "normal":
            continue
        masks = sorted(img.parent.glob(img.stem + "_mask*.png"))
        if masks:
            pairs.append((str(img), [str(m) for m in masks]))
    random.Random(42).shuffle(pairs)
    return pairs


class SegDS(Dataset):
    def __init__(self, pairs, aug=None):
        self.pairs, self.aug = pairs, aug
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mps = self.pairs[i]
        img = cv2.resize(cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB), (SIZE, SIZE))
        m = np.zeros((SIZE, SIZE), np.uint8)
        for mp in mps:  # OR all masks for this image
            m = np.maximum(m, cv2.resize(cv2.imread(mp, cv2.IMREAD_GRAYSCALE), (SIZE, SIZE)))
        m = (m > 0).astype(np.uint8)
        if self.aug:
            a = self.aug(image=img, mask=m); img, m = a["image"], a["mask"]
        x = (img.astype(np.float32) / 255. - self.mean) / self.std
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float()
        y = torch.from_numpy(np.ascontiguousarray(m)).float()[None]
        return x, y


def dice_coef(pred, tgt, eps=1e-6):
    pred = (pred > 0.5).float(); inter = (pred * tgt).sum((1, 2, 3))
    return ((2 * inter + eps) / (pred.sum((1, 2, 3)) + tgt.sum((1, 2, 3)) + eps)).mean().item()


TRAIN_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15,
                       border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
    A.GaussNoise(p=0.2),
])


def run(model, dl_tr, dl_va, epochs, lr, tag):
    bce = nn.BCEWithLogitsLoss(); dice_loss = smp.losses.DiceLoss(mode="binary")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    best, curve = 0.0, []
    for ep in range(epochs):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(DEV), y.to(DEV); opt.zero_grad()
            with torch.autocast("cuda"):
                out = model(x); loss = bce(out, y) + dice_loss(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        model.eval(); ds = []
        with torch.inference_mode():
            for x, y in dl_va:
                ds.append(dice_coef(torch.sigmoid(model(x.to(DEV))).cpu(), y))
        d = float(np.mean(ds)); curve.append(d); best = max(best, d)
        if d >= best:
            torch.save(model.state_dict(), OUT / f"model_{tag}.pt")
        print(f"[{tag}] epoch {ep+1}/{epochs} val_dice={d:.4f} (best {best:.4f})", flush=True)
    return best, curve


def main():
    pairs = collect_pairs(lesion_only=True)
    n_val = len(pairs) // 5
    va, tr = pairs[:n_val], pairs[n_val:]
    print(f"lesion-only pairs={len(pairs)} train={len(tr)} val={len(va)}")
    dl_tr = DataLoader(SegDS(tr, TRAIN_AUG), 8, shuffle=True, num_workers=6, pin_memory=True)
    dl_va = DataLoader(SegDS(va), 8, num_workers=6, pin_memory=True)

    # --- v1-style baseline on the SAME split (fair comparison): plain U-Net, eff-b0, no strong aug ---
    base = smp.Unet("efficientnet-b0", encoder_weights="imagenet", classes=1).to(DEV)
    dl_tr_base = DataLoader(SegDS(tr, A.HorizontalFlip(p=0.5)), 8, shuffle=True, num_workers=6, pin_memory=True)
    base_best, _ = run(base, dl_tr_base, dl_va, epochs=12, lr=3e-4, tag="baseline")

    # --- v2 improved: U-Net++ / eff-b4 / aug / cosine / 60ep ---
    model = smp.UnetPlusPlus("efficientnet-b4", encoder_weights="imagenet", classes=1).to(DEV)
    v2_best, curve = run(model, dl_tr, dl_va, epochs=60, lr=1e-3, tag="v2")

    print(f"\n==== RESULT ====\nv1-style baseline (lesion-only) Dice = {base_best:.4f}"
          f"\nv2 U-Net++/eff-b4/256px/aug   Dice = {v2_best:.4f}"
          f"\nimprovement = +{v2_best - base_best:.4f}")

    plt.figure(figsize=(6, 4)); plt.plot(range(1, len(curve)+1), curve, "-", color="#dd8452")
    plt.axhline(base_best, ls="--", color="gray", label=f"v1 baseline {base_best:.3f}")
    plt.title(f"BUSI v2 val Dice (best {v2_best:.3f})"); plt.xlabel("epoch"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "dice_v2.png", dpi=90); plt.close()

    # overlays with best v2 model
    model.load_state_dict(torch.load(OUT / "model_v2.pt")); model.eval()
    ds = SegDS(va); fig, axes = plt.subplots(4, 3, figsize=(9, 12))
    for r, idx in enumerate(np.linspace(0, len(ds)-1, 4).astype(int)):
        x, y = ds[idx]
        with torch.inference_mode():
            pred = torch.sigmoid(model(x[None].to(DEV)))[0, 0].cpu().numpy()
        rgb = np.clip(x.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) +
                      np.array([0.485, 0.456, 0.406]), 0, 1)
        axes[r, 0].imshow(rgb); axes[r, 0].set_title("Ultrasound"); axes[r, 0].axis("off")
        axes[r, 1].imshow(rgb); axes[r, 1].imshow(y[0], cmap="Reds", alpha=0.5)
        axes[r, 1].set_title("Ground truth"); axes[r, 1].axis("off")
        axes[r, 2].imshow(rgb); axes[r, 2].imshow(pred > 0.5, cmap="Greens", alpha=0.5)
        axes[r, 2].set_title("Prediction (v2)"); axes[r, 2].axis("off")
    fig.suptitle("BUSI U-Net++ v2 — GT vs prediction"); fig.tight_layout()
    fig.savefig(OUT / "overlays_v2.png", dpi=90); plt.close(fig)
    print("saved figures to", OUT)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high"); main()
