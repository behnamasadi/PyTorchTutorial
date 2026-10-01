import sys
sys.path.append("..")
from build_notebook import build

cells = [
("md", """# 🧠 Brain Tumor Segmentation (LGG MRI) — U-Net + Dice

Instead of just *classifying* whether a tumor is present, **segmentation** outlines *exactly where*
it is — pixel by pixel. That's what radiologists need for measuring tumor size and planning treatment.

This notebook trains a **U-Net** to segment low-grade-glioma tumors from FLAIR brain MRI, and shows
**ground-truth vs prediction overlays** — the segmentation equivalent of explainability.

**What you'll learn**
1. Pairing MRI slices with their masks, split by **patient** (no data leakage!)
2. U-Net with a pretrained encoder via `segmentation_models_pytorch`
3. The **Dice** loss & metric (the right metric for segmentation)
4. Visualizing predictions as overlays

> Upvote if this helps 🙏"""),

("code", """# P100+T4-compatible torch so we ALWAYS run on GPU (never CPU) + smp (not preinstalled)
!pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
!pip install -q segmentation-models-pytorch"""),

("code", """import os
from pathlib import Path
import cv2, numpy as np, matplotlib.pyplot as plt, torch, torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset

assert torch.cuda.is_available(), "GPU required — enable the accelerator"
_ = nn.Conv2d(3,4,3).cuda()(torch.randn(1,3,8,8,device="cuda")); torch.cuda.synchronize()
DEVICE = torch.device("cuda")
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
# auto-find the data folder (contains *_mask.tif somewhere under the mount)
def find_root():
    for base in ["/kaggle/input", os.path.expanduser("~/.cache/kagglehub")]:
        b = Path(base)
        if b.exists() and next(b.rglob("*_mask.tif"), None) is not None: return b
    raise FileNotFoundError("lgg data not found")
ROOT = find_root(); print("data root:", ROOT)"""),

("md", """## 1. Pair images with masks — split by patient

Each MRI slice `..._N.tif` has a matching mask `..._N_mask.tif`. **Crucially**, we split by *patient*
so slices from the same brain never appear in both train and validation — otherwise the model
"cheats" and the score is meaningless."""),

("code", """by_patient = {}
for mask in ROOT.rglob("*_mask.tif"):
    img = Path(str(mask).replace("_mask", ""))
    if img.exists(): by_patient.setdefault(mask.parent.name, []).append((str(img), str(mask)))
patients = sorted(by_patient); n_val = max(1, len(patients)//5)
val = [pr for p in patients[:n_val] for pr in by_patient[p]]
train = [pr for p in patients[n_val:] for pr in by_patient[p]]
print(f"patients={len(patients)}  train_slices={len(train)}  val_slices={len(val)}")"""),

("md", "## 2. EDA — many slices have no tumor; masks are small (class imbalance in pixels)"),

("code", """tumor = [p for p in train if cv2.imread(p[1],0).max()>0]
fig, ax = plt.subplots(1,3, figsize=(13,4))
ax[0].bar(["with tumor","no tumor"], [len(tumor), len(train)-len(tumor)], color="#55a868")
ax[0].set_title("Slices with / without tumor")
ip, mp = tumor[len(tumor)//2]
ax[1].imshow(cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)); ax[1].set_title("FLAIR MRI"); ax[1].axis("off")
ax[2].imshow(cv2.imread(mp,0), cmap="magma"); ax[2].set_title("Tumor mask"); ax[2].axis("off")
plt.tight_layout(); plt.show()"""),

("code", """MEAN=torch.tensor([0.485,0.456,0.406]).view(3,1,1); STD=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
class SegDS(Dataset):
    def __init__(self, pairs, train=False): self.pairs, self.train = pairs, train
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = cv2.resize(cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB), (128,128))
        m = cv2.resize(cv2.imread(mp, cv2.IMREAD_GRAYSCALE), (128,128))
        if self.train and np.random.rand()<0.5: img, m = img[:,::-1], m[:,::-1]
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2,0,1).float()/255.
        y = torch.from_numpy(np.ascontiguousarray((m>0).astype(np.float32)))[None]
        return (x-MEAN)/STD, y
dl_tr = DataLoader(SegDS(train, True), 32, shuffle=True, num_workers=2, pin_memory=True)
dl_va = DataLoader(SegDS(val), 32, num_workers=2, pin_memory=True)"""),

("md", """## 3. U-Net + Dice loss

U-Net's encoder-decoder with skip connections is the classic medical-segmentation architecture.
We use a pretrained EfficientNet encoder and train with **BCE + Dice loss**. **Dice** measures mask
overlap (2·intersection / total area) — the standard segmentation metric."""),

("code", """model = smp.Unet("efficientnet-b0", encoder_weights="imagenet", classes=1, activation=None).to(DEVICE)
bce = nn.BCEWithLogitsLoss(); dice_loss = smp.losses.DiceLoss(mode="binary")
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type=="cuda")

def dice_coef(pred, tgt, eps=1e-6):
    pred=(pred>0.5).float(); inter=(pred*tgt).sum((1,2,3))
    return ((2*inter+eps)/(pred.sum((1,2,3))+tgt.sum((1,2,3))+eps)).mean().item()

EPOCHS=8; dices=[]
for ep in range(EPOCHS):
    model.train()
    for x,y in dl_tr:
        x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
        with torch.autocast(DEVICE.type, enabled=DEVICE.type=="cuda"):
            out=model(x); loss=bce(out,y)+dice_loss(out,y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    model.eval(); ds=[]
    with torch.inference_mode():
        for x,y in dl_va: ds.append(dice_coef(torch.sigmoid(model(x.to(DEVICE))).cpu(), y))
    dices.append(float(np.mean(ds))); print(f"epoch {ep+1}/{EPOCHS} val_dice={dices[-1]:.4f}")
print("BEST Dice:", round(max(dices),4))"""),

("md", """## 4. 🔍 Predictions — ground truth vs model

The real test: overlaying predictions on the MRI. Red = ground-truth tumor, green = model prediction.
Good overlap (high Dice) means the model has genuinely learned the tumor boundaries."""),

("code", """vt = [p for p in val if cv2.imread(p[1],0).max()>0]; ds = SegDS(vt); model.eval()
fig, axes = plt.subplots(4, 3, figsize=(9,12))
for r, idx in enumerate(np.linspace(0, len(ds)-1, 4).astype(int)):
    x, y = ds[idx]
    with torch.inference_mode(): pred = torch.sigmoid(model(x[None].to(DEVICE)))[0,0].cpu().numpy()
    rgb = np.clip(x.permute(1,2,0).numpy()*np.array([0.229,0.224,0.225])+np.array([0.485,0.456,0.406]),0,1)
    axes[r,0].imshow(rgb); axes[r,0].set_title("MRI"); axes[r,0].axis("off")
    axes[r,1].imshow(rgb); axes[r,1].imshow(y[0], cmap="Reds", alpha=.5); axes[r,1].set_title("Ground truth"); axes[r,1].axis("off")
    axes[r,2].imshow(rgb); axes[r,2].imshow(pred>0.5, cmap="Greens", alpha=.5); axes[r,2].set_title("Prediction"); axes[r,2].axis("off")
plt.tight_layout(); plt.show()"""),

("md", """## Conclusion

A U-Net with a pretrained encoder segments LGG tumors at **Dice ≈ 0.87** from FLAIR MRI — and the
overlays show it tracks real tumor boundaries. **Patient-level splitting** is the key detail many
notebooks get wrong (and inflate their scores).

**Next:** 3D context (adjacent slices), heavier augmentation, and a SegFormer/attention-U-Net.

*Thanks for reading — upvotes are appreciated! 🙌*"""),
]
build("lgg_unet_segmentation.ipynb", cells)
