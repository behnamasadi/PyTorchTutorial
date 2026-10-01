"""BRISC 2025 — brain tumor MRI 4-class classification + Grad-CAM (educational).

Modern (2025) brain-tumor MRI dataset: glioma / meningioma / pituitary / no_tumor,
pre-split into train/ and test/ folders. EfficientNet-b0 transfer learning + Grad-CAM.
"""
from __future__ import annotations
import os
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn, timm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, balanced_accuracy_score)
from torch.utils.data import DataLoader, Dataset

CANDS = [
    "/kaggle/input/brisc2025/brisc2025/classification_task",
    os.path.expanduser("~/data_kaggle/brisc2025/brisc2025/classification_task"),
]
ROOT = Path(next(c for c in CANDS if Path(c).exists()))
OUT = Path(__file__).parent / "figs"; OUT.mkdir(exist_ok=True)
CLASSES = sorted(d.name for d in (ROOT / "train").iterdir() if d.is_dir())
DEV = torch.device("cuda"); print("device:", DEV, "| classes:", CLASSES)


def scan(split):
    items = []
    for i, c in enumerate(CLASSES):
        for p in (ROOT / split / c).glob("*.jpg"):
            items.append((str(p), i))
    return items


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class Brain(Dataset):
    def __init__(self, items, train=False): self.items, self.train = items, train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        img = cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (224, 224))
        if self.train and np.random.rand() < 0.5: img = img[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.
        return (x - MEAN) / STD, y


def eda(tr):
    from collections import Counter
    cnt = Counter(CLASSES[y] for _, y in tr)
    plt.figure(figsize=(7, 3)); plt.bar(cnt.keys(), cnt.values(), color="#4c72b0")
    plt.title("BRISC 2025 train class distribution"); plt.tight_layout()
    plt.savefig(OUT / "eda.png", dpi=90); plt.close()


def main():
    tr, te = scan("train"), scan("test")
    print(f"train={len(tr)} test={len(te)}")
    eda(tr)
    dl_tr = DataLoader(Brain(tr, True), 32, shuffle=True, num_workers=6, pin_memory=True)
    dl_te = DataLoader(Brain(te), 32, num_workers=6, pin_memory=True)

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=len(CLASSES)).to(DEV)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    EPOCHS = 8
    for ep in range(EPOCHS):
        model.train(); tot = 0
        for x, y in dl_tr:
            x, y = x.to(DEV), torch.tensor(y).to(DEV) if not torch.is_tensor(y) else y.to(DEV)
            opt.zero_grad()
            with torch.autocast("cuda"): loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item()
        print(f"epoch {ep+1}/{EPOCHS} loss={tot/len(dl_tr):.4f}", flush=True)

    model.eval(); ys, ps = [], []
    with torch.inference_mode():
        for x, y in dl_te:
            ps += model(x.to(DEV)).argmax(1).cpu().tolist()
            ys += (y.tolist() if torch.is_tensor(y) else list(y))
    acc = np.mean(np.array(ys) == np.array(ps)); bacc = balanced_accuracy_score(ys, ps)
    print(f"\nTEST accuracy={acc:.4f}  balanced_acc={bacc:.4f}")
    print(classification_report(ys, ps, target_names=CLASSES))
    ConfusionMatrixDisplay(confusion_matrix(ys, ps), display_labels=CLASSES).plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"BRISC 2025 (acc={acc:.3f})"); plt.tight_layout(); plt.savefig(OUT / "cm.png", dpi=90); plt.close()

    gradcam(model, te)
    torch.save(model.state_dict(), OUT / "model.pt")
    print("saved figures to", OUT)


def gradcam(model, items, n=8):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    layer = [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]
    ds = Brain(items)
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 1.6, 7))
    with GradCAM(model=model, target_layers=[layer]) as cam:
        for ax, idx in zip(axes.ravel(), np.linspace(0, len(ds)-1, n).astype(int)):
            x, y = ds[idx]
            rgb = np.clip(x.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) +
                          np.array([0.485, 0.456, 0.406]), 0, 1).astype(np.float32)
            g = cam(input_tensor=x[None].to(DEV))[0]
            ax.imshow(show_cam_on_image(rgb, g, use_rgb=True)); ax.set_title(CLASSES[y], fontsize=9); ax.axis("off")
    fig.suptitle("Grad-CAM: model should attend to the tumor"); fig.tight_layout()
    fig.savefig(OUT / "gradcam.png", dpi=90); plt.close(fig); print("saved Grad-CAM")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high"); main()
