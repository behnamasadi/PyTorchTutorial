import sys
sys.path.append("..")
from build_notebook import build

INSTALL = """!pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
!pip install -q segmentation-models-pytorch"""

SETUP = """import os, random
from pathlib import Path
import cv2, numpy as np, matplotlib.pyplot as plt, torch, torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset

assert torch.cuda.is_available(), "GPU required"
_ = nn.Conv2d(3,4,3).cuda()(torch.randn(1,3,8,8,device="cuda")); torch.cuda.synchronize()
DEVICE = torch.device("cuda"); print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))

def find_root():
    for base in ["/kaggle/input", os.path.expanduser("~/.cache/kagglehub")]:
        b = Path(base)
        if b.exists() and next(b.rglob("*_mask.png"), None) is not None: return b
    raise FileNotFoundError("BUSI data not found")
ROOT = find_root(); print("data root:", ROOT)"""

DATA = """def collect():
    pairs=[]
    for img in ROOT.rglob("*.png"):
        if "_mask" in img.name: continue
        m = img.with_name(img.stem + "_mask.png")
        if m.exists(): pairs.append((str(img), str(m)))
    random.Random(42).shuffle(pairs); return pairs
pairs = collect(); n_val=len(pairs)//5; val, train = pairs[:n_val], pairs[n_val:]
print(f"pairs={len(pairs)} train={len(train)} val={len(val)}")
# EDA: a lesion + its mask
ip, mp = next(p for p in train if "malignant" in p[0])
fig,ax=plt.subplots(1,2,figsize=(9,4))
ax[0].imshow(cv2.cvtColor(cv2.imread(ip),cv2.COLOR_BGR2RGB)); ax[0].set_title("Ultrasound"); ax[0].axis("off")
ax[1].imshow(cv2.imread(mp,0),cmap="magma"); ax[1].set_title("Lesion mask"); ax[1].axis("off"); plt.show()"""

DATASET = """MEAN=torch.tensor([0.485,0.456,0.406]).view(3,1,1); STD=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
class SegDS(Dataset):
    def __init__(self, pairs, train=False): self.pairs,self.train=pairs,train
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        ip,mp=self.pairs[i]
        img=cv2.resize(cv2.cvtColor(cv2.imread(ip),cv2.COLOR_BGR2RGB),(128,128))
        m=cv2.resize(cv2.imread(mp,cv2.IMREAD_GRAYSCALE),(128,128))
        if self.train and np.random.rand()<0.5: img,m=img[:,::-1],m[:,::-1]
        x=torch.from_numpy(np.ascontiguousarray(img)).permute(2,0,1).float()/255.
        y=torch.from_numpy(np.ascontiguousarray((m>0).astype(np.float32)))[None]
        return (x-MEAN)/STD, y
dl_tr=DataLoader(SegDS(train,True),16,shuffle=True,num_workers=2,pin_memory=True)
dl_va=DataLoader(SegDS(val),16,num_workers=2,pin_memory=True)"""

TRAIN = """model=smp.Unet("efficientnet-b0",encoder_weights="imagenet",classes=1,activation=None).to(DEVICE)
bce=nn.BCEWithLogitsLoss(); dloss=smp.losses.DiceLoss(mode="binary")
opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5); scaler=torch.cuda.amp.GradScaler()
def dice(pred,tgt,eps=1e-6):
    pred=(pred>0.5).float(); inter=(pred*tgt).sum((1,2,3))
    return ((2*inter+eps)/(pred.sum((1,2,3))+tgt.sum((1,2,3))+eps)).mean().item()
for ep in range(12):
    model.train()
    for x,y in dl_tr:
        x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
        with torch.autocast("cuda"): out=model(x); loss=bce(out,y)+dloss(out,y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    model.eval(); ds=[]
    with torch.inference_mode():
        for x,y in dl_va: ds.append(dice(torch.sigmoid(model(x.to(DEVICE))).cpu(),y))
    print(f"epoch {ep+1}/12 val_dice={np.mean(ds):.4f}")"""

OVERLAY = """vt=[p for p in val if cv2.imread(p[1],0).max()>0]; ds=SegDS(vt); model.eval()
fig,axes=plt.subplots(4,3,figsize=(9,12))
for r,idx in enumerate(np.linspace(0,len(ds)-1,4).astype(int)):
    x,y=ds[idx]
    with torch.inference_mode(): pred=torch.sigmoid(model(x[None].to(DEVICE)))[0,0].cpu().numpy()
    rgb=np.clip(x.permute(1,2,0).numpy()*np.array([0.229,0.224,0.225])+np.array([0.485,0.456,0.406]),0,1)
    axes[r,0].imshow(rgb); axes[r,0].set_title("Ultrasound"); axes[r,0].axis("off")
    axes[r,1].imshow(rgb); axes[r,1].imshow(y[0],cmap="Reds",alpha=.5); axes[r,1].set_title("Ground truth"); axes[r,1].axis("off")
    axes[r,2].imshow(rgb); axes[r,2].imshow(pred>0.5,cmap="Greens",alpha=.5); axes[r,2].set_title("Prediction"); axes[r,2].axis("off")
plt.suptitle("U-Net breast-lesion segmentation — GT vs prediction"); plt.tight_layout(); plt.show()"""

cells = [
("md", """# 🩺 Breast Ultrasound Lesion Segmentation (BUSI) — U-Net + Dice

Segments breast lesions from ultrasound — a noisy, low-contrast modality where segmentation is genuinely
hard. U-Net (pretrained encoder) + Dice loss, with **ground-truth vs prediction overlays**. GPU-only. *Upvotes appreciated 🙏*"""),
("code", INSTALL),
("md", "## Setup — force GPU + auto-locate data"),
("code", SETUP),
("md", "## 1. Pair images with masks"),
("code", DATA),
("code", DATASET),
("md", "## 2. Train U-Net (BCE + Dice loss)"),
("code", TRAIN),
("md", "## 3. 🔍 Predictions — ground truth vs model"),
("code", OVERLAY),
("md", "## Conclusion\\nU-Net segments breast-ultrasound lesions despite the noise. Ultrasound is harder than MRI (lower Dice) — a good, honest teaching case. Next: attention U-Net, heavier augmentation, post-processing. *Thanks — upvotes welcome! 🙌*"),
]
build("busi_unet_segmentation.ipynb", cells)
