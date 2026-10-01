"""APTOS DR training — Lightning + timm ordinal classifier with CLAHE + Grad-CAM.

Reuses ../common/lit_timm.py. Runnable once the data is downloaded:
    kaggle competitions download -c aptos2019-blindness-detection -p ./data && unzip -q ./data/*.zip -d ./data
    python train.py --data ./data --folds 5 --epochs 20
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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping  # noqa: E402
import albumentations as A  # noqa: E402
from albumentations.pytorch import ToTensorV2  # noqa: E402


def circle_crop_clahe(img: np.ndarray, size: int) -> np.ndarray:
    """Ben-Graham style: crop the fundus circle + CLAHE on the L channel."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > gray.mean() / 3
    coords = np.argwhere(mask)
    if coords.size:
        y0, x0 = coords.min(0); y1, x1 = coords.max(0)
        img = img[y0:y1, x0:x1]
    img = cv2.resize(img, (size, size))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class AptosDataset(Dataset):
    def __init__(self, df, img_dir: Path, size: int, train: bool):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.size = size
        aug = [A.HorizontalFlip(), A.VerticalFlip(), A.ShiftScaleRotate(rotate_limit=25),
               A.RandomBrightnessContrast()] if train else []
        self.tf = A.Compose([*aug,
                             A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                             ToTensorV2()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(str(self.img_dir / f"{row.id_code}.png")), cv2.COLOR_BGR2RGB)
        img = circle_crop_clahe(img, self.size)
        return self.tf(image=img)["image"], int(row.diagnosis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--backbone", default="tf_efficientnetv2_s.in21k_ft_in1k")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    data = Path(args.data)
    df = pd.read_csv(data / "train.csv")
    img_dir = data / "train_images"
    skf = StratifiedKFold(args.folds, shuffle=True, random_state=42)

    for fold, (tr, va) in enumerate(skf.split(df, df.diagnosis)):
        dl_tr = DataLoader(AptosDataset(df.iloc[tr], img_dir, args.size, True),
                           batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        dl_va = DataLoader(AptosDataset(df.iloc[va], img_dir, args.size, False),
                           batch_size=args.bs, num_workers=4, pin_memory=True)
        model = TimmClassifier(args.backbone, num_classes=5, ordinal=True)
        trainer = pl.Trainer(
            max_epochs=args.epochs, precision="16-mixed", accelerator="gpu", devices=1,
            callbacks=[ModelCheckpoint(monitor="val/qwk", mode="max",
                                       filename=f"fold{fold}-{{val/qwk:.4f}}"),
                       EarlyStopping(monitor="val/qwk", mode="max", patience=5)],
        )
        trainer.fit(model, dl_tr, dl_va)
        print(f"[fold {fold}] best QWK = {trainer.checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
