"""APTOS DR training on the public 2019 mirror (no competition join needed).

Trains the reusable Lightning + timm ordinal classifier and emits QWK + a Grad-CAM
montage — proves the B pipeline end-to-end on real retinal data while the real
competition data is gated on the join.

    python train_public.py --data <kagglehub_path> --epochs 8 --folds 1

Dataset layout: train.csv (id_code,diagnosis) + colored_images/<Class>/<id_code>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
from lit_timm import TimmClassifier  # noqa: E402

import pytorch_lightning as pl  # noqa: E402
from pytorch_lightning.callbacks import ModelCheckpoint  # noqa: E402
import albumentations as A  # noqa: E402
from albumentations.pytorch import ToTensorV2  # noqa: E402


def crop_fundus(img):
    """Tight-crop the retinal disc + circular mask to kill the black-corner shortcut."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    coords = np.argwhere(gray > 7)
    if len(coords):
        y0, x0 = coords.min(0); y1, x1 = coords.max(0)
        img = img[y0:y1 + 1, x0:x1 + 1]
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(0.5 * min(h, w)), 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)


def clahe_rgb(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class DS(Dataset):
    def __init__(self, df, size, train):
        self.df = df.reset_index(drop=True)
        aug = [A.HorizontalFlip(), A.VerticalFlip(), A.ShiftScaleRotate(rotate_limit=25),
               A.RandomBrightnessContrast()] if train else []
        self.tf = A.Compose([A.Resize(size, size), *aug,
                             A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                             ToTensorV2()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB)
        img = clahe_rgb(crop_fundus(img))
        return self.tf(image=img)["image"], int(r["diagnosis"])


def build_df(data: Path):
    df = pd.read_csv(data / "train.csv")
    paths = {p.stem: str(p) for p in (data / "colored_images").rglob("*.png")}
    df["path"] = df.id_code.map(paths)
    return df.dropna(subset=["path"]).reset_index(drop=True)


def gradcam_montage(model, ds, out_png, n=8):
    """Save a Grad-CAM overlay grid for n validation samples."""
    sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
    from gradcam import overlay_cam
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = model.cuda().eval()
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 1.5, 6))
    for ax, idx in zip(axes.ravel(), np.linspace(0, len(ds) - 1, n).astype(int)):
        x, y = ds[idx]
        rgb = (x.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) +
               np.array([0.485, 0.456, 0.406]))
        rgb = np.clip(rgb, 0, 1).astype(np.float32)
        heat = overlay_cam(model.net, x[None].cuda(), rgb)
        ax.imshow(heat); ax.set_title(f"gt={y}", fontsize=8); ax.axis("off")
    fig.suptitle("Grad-CAM — APTOS DR (should attend to lesions)")
    fig.tight_layout(); fig.savefig(out_png, dpi=90)
    print(f"[aptos] saved Grad-CAM montage -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--backbone", default="tf_efficientnetv2_s.in21k_ft_in1k")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--folds", type=int, default=1)
    args = ap.parse_args()

    df = build_df(Path(args.data))
    print(f"[aptos] {len(df)} images, class dist: {df.diagnosis.value_counts().to_dict()}")
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    fold_qwks, last_model, last_va = [], None, None
    for fold, (tr, va) in enumerate(skf.split(df, df.diagnosis)):
        if fold >= args.folds:
            break
        dl_tr = DataLoader(DS(df.iloc[tr], args.size, True), batch_size=args.bs,
                           shuffle=True, num_workers=4, pin_memory=True)
        dl_va = DataLoader(DS(df.iloc[va], args.size, False), batch_size=args.bs,
                           num_workers=4, pin_memory=True)
        model = TimmClassifier(args.backbone, num_classes=5, ordinal=True)
        ckpt = ModelCheckpoint(monitor="val/qwk", mode="max", dirpath=f"work_aptos/f{fold}",
                               filename="best-{val/qwk:.4f}")
        trainer = pl.Trainer(max_epochs=args.epochs, precision="16-mixed",
                             accelerator="gpu", devices=1, callbacks=[ckpt],
                             enable_progress_bar=False, logger=False)
        trainer.fit(model, dl_tr, dl_va)
        q = float(ckpt.best_model_score)
        fold_qwks.append(q)
        print(f"[aptos] fold {fold} best val QWK = {q:.4f}", flush=True)
        last_model, last_va = model, DS(df.iloc[va], args.size, False)
    print(f"[aptos] {args.folds}-fold QWK: {[round(q,4) for q in fold_qwks]}  "
          f"MEAN = {np.mean(fold_qwks):.4f} ± {np.std(fold_qwks):.4f}")
    gradcam_montage(last_model, last_va, "work_aptos/gradcam.png")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
