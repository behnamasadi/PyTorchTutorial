"""Chest X-Ray Pneumonia — EfficientNet + Grad-CAM (educational).

Engine for the published Kaggle notebook. Runs locally on the 3090 (and on Kaggle
via safe_device fallback). Produces every figure the notebook renders:
EDA, training curves, confusion matrix, ROC, and Grad-CAM overlays.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report, confusion_matrix, roc_auc_score)
from torch.utils.data import DataLoader, Dataset

# ---- data location (works locally + on Kaggle) ------------------------------
CANDS = [
    "/kaggle/input/chest-xray-pneumonia/chest_xray",
    os.path.expanduser("~/.cache/kagglehub/datasets/paultimothymooney/"
                       "chest-xray-pneumonia/versions/2/chest_xray"),
]
ROOT = Path(next(c for c in CANDS if Path(c).exists()))
OUT = Path("figs"); OUT.mkdir(exist_ok=True)
CLASSES = ["NORMAL", "PNEUMONIA"]


def safe_device():
    """P100-safe device pick — Kaggle's torch may lack the assigned GPU's arch."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        _ = nn.Conv2d(3, 4, 3).cuda()(torch.randn(1, 3, 8, 8, device="cuda"))
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception as e:
        print("GPU unusable, CPU:", str(e)[:70]); return torch.device("cpu")


DEV = safe_device()
print("device:", DEV)


def list_split(split):
    rows = []
    for lbl, cls in enumerate(CLASSES):
        for f in (ROOT / split / cls).glob("*.jpeg"):
            rows.append((str(f), lbl))
    return rows


class XRay(Dataset):
    def __init__(self, rows, size=224, train=False):
        self.rows, self.size, self.train = rows, size, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, lbl = self.rows[i]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        if self.train and np.random.rand() < 0.5:
            img = img[:, ::-1]  # horizontal flip
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (x - mean) / std, lbl


def eda(train_rows):
    counts = {c: sum(1 for _, l in train_rows if l == i) for i, c in enumerate(CLASSES)}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(counts.keys(), counts.values(), color=["#4c72b0", "#c44e52"])
    ax[0].set_title("Train class distribution (imbalanced)")
    for i, c in enumerate(CLASSES):  # one sample of each class
        p = next(r[0] for r in train_rows if r[1] == i)
        ax[1].imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY), cmap="gray")
    ax[1].set_title("Sample X-ray"); ax[1].axis("off")
    fig.tight_layout(); fig.savefig(OUT / "eda.png", dpi=90); plt.close(fig)
    return counts


def main():
    tr, va, te = list_split("train"), list_split("val"), list_split("test")
    print(f"train={len(tr)} val={len(va)} test={len(te)}")
    counts = eda(tr)

    dl_tr = DataLoader(XRay(tr, train=True), 32, shuffle=True, num_workers=4, pin_memory=True)
    dl_te = DataLoader(XRay(te), 32, num_workers=4, pin_memory=True)

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2).to(DEV)
    # class-weighted loss to handle 1341 vs 3875 imbalance
    w = torch.tensor([len(tr) / counts["NORMAL"], len(tr) / counts["PNEUMONIA"]]).float().to(DEV)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=DEV.type == "cuda")

    hist = []
    EPOCHS = 4
    for ep in range(EPOCHS):
        model.train(); tot = 0
        for x, y in dl_tr:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        hist.append(tot / len(dl_tr))
        print(f"epoch {ep+1}/{EPOCHS} loss={hist[-1]:.4f}", flush=True)

    # evaluate
    model.eval(); ys, ps, probs = [], [], []
    with torch.inference_mode():
        for x, y in dl_te:
            logit = model(x.to(DEV))
            pr = torch.softmax(logit, 1)[:, 1].cpu().numpy()
            ps += logit.argmax(1).cpu().tolist(); ys += y.tolist(); probs += pr.tolist()
    acc = np.mean(np.array(ys) == np.array(ps)); auc = roc_auc_score(ys, probs)
    print(f"\nTEST accuracy={acc:.4f}  AUC={auc:.4f}")
    print(classification_report(ys, ps, target_names=CLASSES))

    # training curve
    plt.figure(figsize=(6, 4)); plt.plot(range(1, EPOCHS+1), hist, "o-")
    plt.title("Training loss"); plt.xlabel("epoch"); plt.savefig(OUT / "loss.png", dpi=90); plt.close()
    # confusion matrix
    ConfusionMatrixDisplay(confusion_matrix(ys, ps), display_labels=CLASSES).plot(cmap="Blues")
    plt.title(f"Confusion Matrix (acc={acc:.3f})"); plt.savefig(OUT / "cm.png", dpi=90); plt.close()
    RocCurveDisplay.from_predictions(ys, probs); plt.title(f"ROC (AUC={auc:.3f})")
    plt.savefig(OUT / "roc.png", dpi=90); plt.close()

    gradcam(model, te)
    torch.save(model.state_dict(), OUT / "model.pt")
    print("saved all figures to", OUT)


def gradcam(model, test_rows, n=6):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    layer = [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]
    ds = XRay(test_rows)
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 2, 7))
    idxs = list(np.linspace(0, len(ds) - 1, n).astype(int))
    with GradCAM(model=model, target_layers=[layer]) as cam:
        for ax, idx in zip(axes.ravel(), idxs):
            x, y = ds[idx]
            rgb = (x.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) +
                   np.array([0.485, 0.456, 0.406]))
            rgb = np.clip(rgb, 0, 1).astype(np.float32)
            g = cam(input_tensor=x[None].to(DEV))[0]
            ax.imshow(show_cam_on_image(rgb, g, use_rgb=True))
            ax.set_title(f"true: {CLASSES[y]}", fontsize=9); ax.axis("off")
    fig.suptitle("Grad-CAM — where the model looks (should focus on lung opacities)")
    fig.tight_layout(); fig.savefig(OUT / "gradcam.png", dpi=90); plt.close(fig)
    print("saved Grad-CAM")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
