"""Skin Lesion (HAM10000) — 7-class classification + Grad-CAM (educational).

Classifies 7 skin-lesion types (incl. melanoma) from dermatoscopy. Highlights
*class imbalance* handling (nv dominates) with a weighted loss, and Grad-CAM to
show the model attends to the lesion. Runs on the 3090 (Kaggle via safe_device).
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

CANDS = [
    "/kaggle/input/skin-cancer-mnist-ham10000",
    os.path.expanduser("~/.cache/kagglehub/datasets/kmader/"
                       "skin-cancer-mnist-ham10000/versions/2"),
]
ROOT = Path(next(c for c in CANDS if Path(c).exists()))
OUT = Path("figs"); OUT.mkdir(exist_ok=True)
# 7 diagnoses; full names for readability
DX = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def safe_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        _ = nn.Conv2d(3, 4, 3).cuda()(torch.randn(1, 3, 8, 8, device="cuda"))
        torch.cuda.synchronize(); return torch.device("cpu" if False else "cuda")
    except Exception as e:
        print("GPU unusable, CPU:", str(e)[:70]); return torch.device("cpu")


DEV = safe_device(); print("device:", DEV)


def load_meta():
    csv = next(ROOT.rglob("HAM10000_metadata*.csv"))
    df = pd.read_csv(csv)
    paths = {p.stem: str(p) for p in ROOT.rglob("*.jpg")}
    df["path"] = df.image_id.map(paths)
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    df["label"] = df.dx.map({d: i for i, d in enumerate(DX)})
    return df


class Skin(Dataset):
    def __init__(self, df, size=224, train=False):
        self.df, self.size, self.train = df.reset_index(drop=True), size, train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.resize(cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB), (self.size, self.size))
        if self.train and np.random.rand() < 0.5:
            img = img[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (x - mean) / std, int(r["label"])


def eda(df):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    vc = df.dx.value_counts()
    ax[0].bar(vc.index, vc.values, color="#c44e52")
    ax[0].set_title("Class distribution (severe imbalance: nv dominates)")
    grid = np.zeros((150, 150 * 7, 3), np.uint8)
    for i, d in enumerate(DX):
        p = df[df.dx == d].iloc[0]["path"]
        grid[:, i*150:(i+1)*150] = cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (150, 150))
    ax[1].imshow(grid); ax[1].set_title("Samples: " + " ".join(DX)); ax[1].axis("off")
    fig.tight_layout(); fig.savefig(OUT / "eda.png", dpi=90); plt.close(fig)


def main():
    df = load_meta()
    print(f"images={len(df)} classes={df.dx.nunique()}")
    eda(df)
    tr_df, te_df = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)
    dl_tr = DataLoader(Skin(tr_df, train=True), 32, shuffle=True, num_workers=4, pin_memory=True)
    dl_te = DataLoader(Skin(te_df), 32, num_workers=4, pin_memory=True)

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=len(DX)).to(DEV)
    # inverse-frequency class weights for the imbalance
    freq = tr_df.label.value_counts().sort_index().values
    w = torch.tensor(freq.sum() / (len(DX) * freq)).float().to(DEV)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=DEV.type == "cuda")

    hist, EPOCHS = [], 6
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
    from sklearn.metrics import balanced_accuracy_score
    bacc = balanced_accuracy_score(ys, ps)
    print(f"\nTEST accuracy={acc:.4f}  balanced_acc={bacc:.4f}")
    print(classification_report(ys, ps, target_names=DX))

    ConfusionMatrixDisplay(confusion_matrix(ys, ps), display_labels=DX).plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix (bal-acc={bacc:.3f})"); plt.tight_layout()
    plt.savefig(OUT / "cm.png", dpi=90); plt.close()

    gradcam(model, te_df)
    torch.save(model.state_dict(), OUT / "model.pt")
    print("saved all figures to", OUT)


def gradcam(model, df, n=8):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    layer = [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]
    ds = Skin(df.reset_index(drop=True))
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 1.6, 7))
    with GradCAM(model=model, target_layers=[layer]) as cam:
        for ax, idx in zip(axes.ravel(), np.linspace(0, len(ds)-1, n).astype(int)):
            x, y = ds[idx]
            rgb = np.clip(x.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) +
                          np.array([0.485, 0.456, 0.406]), 0, 1).astype(np.float32)
            g = cam(input_tensor=x[None].to(DEV))[0]
            ax.imshow(show_cam_on_image(rgb, g, use_rgb=True))
            ax.set_title(DX[y], fontsize=9); ax.axis("off")
    fig.suptitle("Grad-CAM — model should attend to the lesion")
    fig.tight_layout(); fig.savefig(OUT / "gradcam.png", dpi=90); plt.close(fig)
    print("saved Grad-CAM")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high"); main()
