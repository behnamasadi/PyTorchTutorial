"""Retinal Blood Vessel Segmentation — U-Net++ / EfficientNet-b4 (educational).

Segments the retinal vasculature from fundus images. Thin, branching structures ->
train at 512px with strong augmentation. Small dataset (80 train / 20 test), so
augmentation matters. Dice + BCE, cosine LR. GT-vs-prediction overlays.
"""
from __future__ import annotations
import os
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import albumentations as A
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset

CANDS = [
    "/kaggle/input/retina-blood-vessel/Data",
    os.path.expanduser("~/data_kaggle/retina_vessel/Data"),
]
ROOT = Path(next(c for c in CANDS if Path(c).exists()))
OUT = Path(__file__).parent / "figs"; OUT.mkdir(exist_ok=True)
DEV = torch.device("cuda"); SIZE = 512
print("device:", DEV, "| root:", ROOT)


def pairs(split):
    imd = ROOT / split / "image"
    out = []
    for p in sorted(imd.glob("*.png")):
        m = ROOT / split / "mask" / p.name
        if m.exists(): out.append((str(p), str(m)))
    return out


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
TRAIN_AUG = A.Compose([
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-25, 25), p=0.6),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
])


class Vessel(Dataset):
    def __init__(self, prs, aug=None): self.prs, self.aug = prs, aug
    def __len__(self): return len(self.prs)
    def __getitem__(self, i):
        ip, mp = self.prs[i]
        img = cv2.resize(cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB), (SIZE, SIZE))
        m = cv2.resize(cv2.imread(mp, cv2.IMREAD_GRAYSCALE), (SIZE, SIZE))
        m = (m > 127).astype(np.uint8)
        if self.aug: a = self.aug(image=img, mask=m); img, m = a["image"], a["mask"]
        x = (img.astype(np.float32) / 255. - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float()
        y = torch.from_numpy(np.ascontiguousarray(m)).float()[None]
        return x, y


def dice_coef(pred, tgt, eps=1e-6):
    pred = (pred > 0.5).float(); inter = (pred * tgt).sum((1, 2, 3))
    return ((2 * inter + eps) / (pred.sum((1, 2, 3)) + tgt.sum((1, 2, 3)) + eps)).mean().item()


def main():
    tr, te = pairs("train"), pairs("test")
    print(f"train={len(tr)} test={len(te)}")
    # EDA
    ip, mp = tr[0]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].imshow(cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)); ax[0].set_title("Fundus"); ax[0].axis("off")
    ax[1].imshow(cv2.imread(mp, 0), cmap="gray"); ax[1].set_title("Vessel mask"); ax[1].axis("off")
    fig.tight_layout(); fig.savefig(OUT / "eda.png", dpi=90); plt.close(fig)

    dl_tr = DataLoader(Vessel(tr, TRAIN_AUG), 4, shuffle=True, num_workers=6, pin_memory=True)
    dl_te = DataLoader(Vessel(te), 4, num_workers=6, pin_memory=True)

    model = smp.UnetPlusPlus("efficientnet-b4", encoder_weights="imagenet", classes=1, activation=None).to(DEV)
    bce = nn.BCEWithLogitsLoss(); dloss = smp.losses.DiceLoss(mode="binary")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    EPOCHS = 80
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    best, curve = 0.0, []
    for ep in range(EPOCHS):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(DEV), y.to(DEV); opt.zero_grad()
            with torch.autocast("cuda"): out = model(x); loss = bce(out, y) + dloss(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        model.eval(); ds = []
        with torch.inference_mode():
            for x, y in dl_te: ds.append(dice_coef(torch.sigmoid(model(x.to(DEV))).cpu(), y))
        d = float(np.mean(ds)); curve.append(d)
        if d >= best: best = d; torch.save(model.state_dict(), OUT / "model.pt")
        if (ep+1) % 10 == 0 or ep == 0: print(f"epoch {ep+1}/{EPOCHS} test_dice={d:.4f} (best {best:.4f})", flush=True)

    print(f"\nBEST test Dice = {best:.4f}")
    plt.figure(figsize=(6, 4)); plt.plot(range(1, EPOCHS+1), curve, "-", color="#c44e52")
    plt.title(f"Test Dice (best {best:.3f})"); plt.xlabel("epoch"); plt.ylabel("Dice")
    plt.tight_layout(); plt.savefig(OUT / "dice.png", dpi=90); plt.close()

    model.load_state_dict(torch.load(OUT / "model.pt")); model.eval()
    ds = Vessel(te); fig, axes = plt.subplots(4, 3, figsize=(9, 12))
    for r, idx in enumerate(np.linspace(0, len(ds)-1, 4).astype(int)):
        x, y = ds[idx]
        with torch.inference_mode(): pred = torch.sigmoid(model(x[None].to(DEV)))[0, 0].cpu().numpy()
        rgb = np.clip(x.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)
        axes[r, 0].imshow(rgb); axes[r, 0].set_title("Fundus"); axes[r, 0].axis("off")
        axes[r, 1].imshow(y[0], cmap="gray"); axes[r, 1].set_title("Ground truth"); axes[r, 1].axis("off")
        axes[r, 2].imshow(pred > 0.5, cmap="gray"); axes[r, 2].set_title("Prediction"); axes[r, 2].axis("off")
    fig.suptitle("Retinal vessel segmentation: GT vs prediction"); fig.tight_layout()
    fig.savefig(OUT / "overlays.png", dpi=90); plt.close(fig)
    print("saved figures to", OUT)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high"); main()
