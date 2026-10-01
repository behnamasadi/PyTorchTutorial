"""Retinal OCT (Kermany) — 4-class classification + Grad-CAM (educational).

Classifies retinal OCT scans into CNV / DME / DRUSEN / NORMAL. Large dataset (~83k),
so we cap train images per class for a fast, reproducible run. Grad-CAM shows the model
attends to the retinal-layer pathology. Runs on 3090 (Kaggle via safe_device).
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
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from torch.utils.data import DataLoader, Dataset

# find the OCT2017 dir (its name has a trailing space in this dataset)
BASE = [
    "/kaggle/input/kermany2018",
    os.path.expanduser("~/.cache/kagglehub/datasets/paultimothymooney/kermany2018/versions/2"),
]
base = Path(next(b for b in BASE if Path(b).exists()))
ROOT = next(d for d in base.rglob("train") if (d / "NORMAL").exists()).parent
OUT = Path("figs"); OUT.mkdir(exist_ok=True)
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CAP = 3000  # max train images per class (speed)


def safe_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        _ = nn.Conv2d(3, 4, 3).cuda()(torch.randn(1, 3, 8, 8, device="cuda"))
        torch.cuda.synchronize(); return torch.device("cuda")
    except Exception as e:
        print("GPU unusable, CPU:", str(e)[:70]); return torch.device("cpu")


DEV = safe_device(); print("device:", DEV)


def list_split(split, cap=None):
    rows = []
    for lbl, cls in enumerate(CLASSES):
        fs = sorted((ROOT / split / cls).glob("*.jpeg"))
        if cap:
            random.Random(42).shuffle(fs); fs = fs[:cap]
        rows += [(str(f), lbl) for f in fs]
    return rows


class OCT(Dataset):
    def __init__(self, rows, size=224, train=False):
        self.rows, self.size, self.train = rows, size, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, lbl = self.rows[i]
        img = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), (self.size, self.size))
        if self.train and np.random.rand() < 0.5:
            img = img[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (x - mean) / std, lbl


def eda(rows):
    counts = [sum(1 for _, l in rows if l == i) for i in range(len(CLASSES))]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].bar(CLASSES, counts, color="#8172b3"); ax[0].set_title(f"Train (capped {CAP}/class)")
    grid = np.zeros((224, 224 * 4, 3), np.uint8)
    for i in range(len(CLASSES)):
        p = next(r[0] for r in rows if r[1] == i)
        grid[:, i*224:(i+1)*224] = cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (224, 224))
    ax[1].imshow(grid); ax[1].set_title("Samples: " + " | ".join(CLASSES)); ax[1].axis("off")
    fig.tight_layout(); fig.savefig(OUT / "eda.png", dpi=90); plt.close(fig)


def main():
    tr, te = list_split("train", CAP), list_split("test")
    print(f"train={len(tr)} test={len(te)}")
    eda(tr)
    dl_tr = DataLoader(OCT(tr, train=True), 64, shuffle=True, num_workers=6, pin_memory=True)
    dl_te = DataLoader(OCT(te), 64, num_workers=4, pin_memory=True)

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=len(CLASSES)).to(DEV)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=DEV.type == "cuda")

    hist, EPOCHS = [], 3
    for ep in range(EPOCHS):
        model.train(); tot = 0
        for x, y in dl_tr:
            x, y = x.to(DEV), y.to(DEV); opt.zero_grad()
            with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item()
        hist.append(tot/len(dl_tr)); print(f"epoch {ep+1}/{EPOCHS} loss={hist[-1]:.4f}", flush=True)

    model.eval(); ys, ps = [], []
    with torch.inference_mode():
        for x, y in dl_te:
            ps += model(x.to(DEV)).argmax(1).cpu().tolist(); ys += y.tolist()
    acc = np.mean(np.array(ys) == np.array(ps))
    print(f"\nTEST accuracy={acc:.4f}")
    print(classification_report(ys, ps, target_names=CLASSES))
    ConfusionMatrixDisplay(confusion_matrix(ys, ps), display_labels=CLASSES).plot(cmap="Blues")
    plt.title(f"Confusion Matrix (acc={acc:.3f})"); plt.tight_layout()
    plt.savefig(OUT / "cm.png", dpi=90); plt.close()

    gradcam(model, te)
    torch.save(model.state_dict(), OUT / "model.pt")
    print("saved all figures to", OUT)


def gradcam(model, rows, n=8):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    layer = [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]
    ds = OCT(rows)
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 1.6, 7))
    with GradCAM(model=model, target_layers=[layer]) as cam:
        for ax, idx in zip(axes.ravel(), np.linspace(0, len(ds)-1, n).astype(int)):
            x, y = ds[idx]
            rgb = np.clip(x.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) +
                          np.array([0.485, 0.456, 0.406]), 0, 1).astype(np.float32)
            g = cam(input_tensor=x[None].to(DEV))[0]
            ax.imshow(show_cam_on_image(rgb, g, use_rgb=True))
            ax.set_title(CLASSES[y], fontsize=9); ax.axis("off")
    fig.suptitle("Grad-CAM — model should attend to the retinal pathology")
    fig.tight_layout(); fig.savefig(OUT / "gradcam.png", dpi=90); plt.close(fig)
    print("saved Grad-CAM")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high"); main()
