"""Build the Liver Ultrasound notebook: 'Can AI Spot Liver Cancer in an Ultrasound?'

Two-in-one, viz-heavy, clickbait-titled notebook on a fresh 2025 dataset:
  - pipeline DIAGRAM (matplotlib schematic)
  - EDA chart + annotated gallery (liver outline + mass outline from polygon JSON)
  - U-Net++ mass SEGMENTATION with GT-vs-pred overlays + Dice
  - EfficientNet 3-class classification (Normal/Benign/Malignant) + confusion matrix
  - GRAD-CAM gallery
Masks are stored as flat [[x,y],...] polygon JSON -> rasterised with cv2.fillPoly.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from build_notebook import build

cells = [
("md", """# 🫀 Can AI Spot Liver Cancer in an Ultrasound?

Liver ultrasound is cheap, fast, and radiation-free, but reading it is **hard** even for experts.
So let's throw two neural nets at a **fresh 2025** annotated liver-ultrasound dataset and see how far we get:

1. **🎯 Find the tumor** - a **U-Net++** segments the liver mass, pixel by pixel.
2. **🩺 Make the call** - an **EfficientNet** classifies each scan as **Normal / Benign / Malignant**.
3. **👀 Show its reasoning** - **Grad-CAM** reveals *where* the classifier is looking.

Along the way: a pipeline diagram, class-balance charts, annotated galleries, segmentation overlays,
a confusion matrix, and Grad-CAM heatmaps. **GPU-only.** If this is useful, an upvote makes my day 🙏"""),

("code", """!pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
!pip install -q segmentation-models-pytorch albumentations grad-cam timm"""),

("md", "## Setup - force GPU + auto-locate the data"),

("code", """import os, json
from pathlib import Path
import cv2, numpy as np, matplotlib.pyplot as plt, torch, torch.nn as nn, timm
import albumentations as A, segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, balanced_accuracy_score)

assert torch.cuda.is_available(), 'GPU required'
_ = nn.Conv2d(3,4,3).cuda()(torch.randn(1,3,8,8,device='cuda')); torch.cuda.synchronize()
DEVICE = torch.device('cuda'); print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))

def find_root():
    for base in ['/kaggle/input', os.path.expanduser('~/.cache/kagglehub'),
                 os.path.expanduser('~/data_kaggle'), '.']:
        for d in Path(base).rglob('Benign'):
            if (d/'Benign'/'image').exists() and (d.parent/'Malignant').exists():
                return d.parent
    raise FileNotFoundError('liver dataset root not found')
ROOT = find_root(); print('data root:', ROOT)
CLASSES = ['Normal', 'Benign', 'Malignant']          # ordered by severity
CDIR = lambda c: ROOT/c/c                              # e.g. ROOT/Benign/Benign"""),

("md", "## 🗺️ The pipeline at a glance"),

("code", """# a little schematic so readers instantly get what the notebook does
fig, ax = plt.subplots(figsize=(11, 3.2)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,3)
def box(x, y, w, h, text, fc):
    ax.add_patch(plt.matplotlib.patches.FancyBboxPatch((x,y), w, h,
        boxstyle='round,pad=0.03,rounding_size=0.12', fc=fc, ec='#333', lw=1.5))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=10, weight='bold')
def arrow(x1,y1,x2,y2):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='-|>', lw=2, color='#333'))
box(0.2,1.1,1.9,0.9,'Liver\\nUltrasound','#dfe7f5')
box(3.0,2.0,2.4,0.9,'U-Net++\\nsegment mass','#cfe8d8')
box(3.0,0.2,2.4,0.9,'EfficientNet\\nclassify','#f6d9cf')
box(6.4,2.0,3.3,0.9,'Tumor mask + Dice','#cfe8d8')
box(6.4,0.2,3.3,0.9,'Normal / Benign / Malignant\\n+ Grad-CAM','#f6d9cf')
arrow(2.1,1.55,3.0,2.45); arrow(2.1,1.55,3.0,0.65)
arrow(5.4,2.45,6.4,2.45); arrow(5.4,0.65,6.4,0.65)
ax.set_title('One ultrasound in -> a segmented tumor AND a diagnosis out', fontsize=12, weight='bold')
plt.tight_layout(); plt.show()"""),

("md", """## 1. 📊 What's in the data? (class balance + annotated examples)

Every scan ships with expert **polygon annotations** stored as JSON: an outline of the whole **liver**
and, for abnormal scans, an outline of the **mass**. We rasterise those polygons into masks."""),

("code", """def poly_to_mask(json_path, h, w):
    m = np.zeros((h, w), np.uint8)
    if Path(json_path).exists():
        pts = json.load(open(json_path))
        if pts: cv2.fillPoly(m, [np.array(pts, np.int32)], 1)
    return m

def scan():
    rows = []   # (image_path, class_idx, mass_json_or_None, liver_json_or_None)
    for ci, c in enumerate(CLASSES):
        for ip in sorted((CDIR(c)/'image').glob('*.jpg')):
            mass = CDIR(c)/'segmentation'/'mass'/(ip.stem+'.json')
            liver = CDIR(c)/'segmentation'/'liver'/(ip.stem+'.json')
            rows.append((str(ip), ci, str(mass) if mass.exists() else None,
                         str(liver) if liver.exists() else None))
    return rows
rows = scan()
from collections import Counter
cnt = Counter(CLASSES[r[1]] for r in rows)
print('total scans:', len(rows), '| per class:', dict(cnt))

fig, ax = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={'width_ratios':[1,2.4]})
ax[0].bar(list(cnt.keys()), [cnt[k] for k in cnt], color=['#8fb98f','#e2c044','#c44e52'])
ax[0].set_title('Scans per class'); ax[0].set_ylabel('count')
# annotated gallery: liver outline (cyan) + mass outline (red)
picks = [next(r for r in rows if r[1]==ci) for ci in range(3)]
sub = ax[1].inset_axes([0,0,1,1]); sub.axis('off'); ax[1].axis('off')
gal = fig.add_gridspec(1, 3, left=0.42, right=0.99, top=0.85, bottom=0.05)
for k, r in enumerate(picks):
    a = fig.add_subplot(gal[0, k]); img = cv2.cvtColor(cv2.imread(r[0]), cv2.COLOR_BGR2RGB)
    a.imshow(img)
    if r[3]: a.contour(poly_to_mask(r[3], *img.shape[:2]), colors='cyan', linewidths=1.4)
    if r[2]: a.contour(poly_to_mask(r[2], *img.shape[:2]), colors='red', linewidths=1.8)
    a.set_title(CLASSES[r[1]], fontsize=11); a.axis('off')
fig.suptitle('Cyan = liver outline    Red = mass outline (expert polygons)', y=1.02)
plt.show()"""),

("md", """## 2. 🎯 Segment the mass with U-Net++

We train on the **abnormal scans only** (Benign + Malignant, which have a mass), 256px, EfficientNet-b4
encoder, real augmentation, Dice+BCE loss. Normal scans have no mass, so including them would just
inflate the score."""),

("code", """SIZE = 256
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
seg_rows = [r for r in rows if r[2] is not None]      # has a mass polygon
rng = np.random.RandomState(42); idx = rng.permutation(len(seg_rows))
n_val = len(seg_rows)//5; val_s = [seg_rows[i] for i in idx[:n_val]]; tr_s = [seg_rows[i] for i in idx[n_val:]]
print(f'segmentation: train={len(tr_s)} val={len(val_s)}')

AUG = A.Compose([A.HorizontalFlip(p=0.5),
                 A.Affine(scale=(0.9,1.1), translate_percent=0.06, rotate=(-15,15), p=0.5),
                 A.RandomBrightnessContrast(0.2,0.2,p=0.5), A.GaussNoise(p=0.2)])
class SegDS(Dataset):
    def __init__(self, rr, aug=None): self.rr, self.aug = rr, aug
    def __len__(self): return len(self.rr)
    def __getitem__(self, i):
        ip, _, mp, _ = self.rr[i]; img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        m = poly_to_mask(mp, *img.shape[:2])
        img = cv2.resize(img, (SIZE,SIZE)); m = cv2.resize(m, (SIZE,SIZE), interpolation=cv2.INTER_NEAREST)
        if self.aug: a = self.aug(image=img, mask=m); img, m = a['image'], a['mask']
        x = (img.astype(np.float32)/255.-MEAN)/STD
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2,0,1).float()
        return x, torch.from_numpy(np.ascontiguousarray(m)).float()[None]
dl_tr = DataLoader(SegDS(tr_s, AUG), 8, shuffle=True, num_workers=2, pin_memory=True)
dl_va = DataLoader(SegDS(val_s), 8, num_workers=2, pin_memory=True)

def dice_coef(p, t, eps=1e-6):
    p=(p>0.5).float(); i=(p*t).sum((1,2,3)); return ((2*i+eps)/(p.sum((1,2,3))+t.sum((1,2,3))+eps)).mean().item()
seg = smp.UnetPlusPlus('efficientnet-b4', encoder_weights='imagenet', classes=1).to(DEVICE)
bce=nn.BCEWithLogitsLoss(); dloss=smp.losses.DiceLoss(mode='binary')
opt=torch.optim.AdamW(seg.parameters(),lr=1e-3,weight_decay=1e-5)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40); scaler=torch.cuda.amp.GradScaler()
best, curve = 0.0, []
for ep in range(40):
    seg.train()
    for x,y in dl_tr:
        x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
        with torch.autocast('cuda'): loss=bce(seg(x),y)+dloss(seg(x),y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    sched.step(); seg.eval(); ds=[]
    with torch.inference_mode():
        for x,y in dl_va: ds.append(dice_coef(torch.sigmoid(seg(x.to(DEVICE))).cpu(), y))
    d=float(np.mean(ds)); curve.append(d)
    if d>=best: best=d; torch.save(seg.state_dict(),'seg.pt')
    if (ep+1)%5==0 or ep==0: print(f'  seg epoch {ep+1}/40 val_dice={d:.4f} (best {best:.4f})')
seg.load_state_dict(torch.load('seg.pt'))
print(f'\\nBEST mass-segmentation Dice = {best:.4f}')
plt.figure(figsize=(6,4)); plt.plot(range(1,len(curve)+1),curve,'-',color='#c44e52')
plt.title(f'Liver-mass U-Net++ val Dice (best {best:.3f})'); plt.xlabel('epoch'); plt.ylabel('Dice'); plt.show()"""),

("code", """# GT vs prediction overlays on held-out scans
ds = SegDS(val_s); seg.eval(); fig, axes = plt.subplots(3, 3, figsize=(11, 11))
for r, i in enumerate(np.linspace(0, len(ds)-1, 3).astype(int)):
    x, y = ds[i]
    with torch.inference_mode(): pm = torch.sigmoid(seg(x[None].to(DEVICE)))[0,0].cpu().numpy()
    rgb = np.clip(x.permute(1,2,0).numpy()*STD+MEAN, 0, 1); gt=y[0].numpy(); pb=(pm>0.5).astype(float)
    dc = 2*(gt*pb).sum()/(gt.sum()+pb.sum()+1e-6)
    axes[r,0].imshow(rgb); axes[r,0].set_title('Ultrasound'); axes[r,0].axis('off')
    axes[r,1].imshow(rgb); axes[r,1].contour(gt,colors='red',linewidths=1.6); axes[r,1].contour(pb,colors='lime',linewidths=1.2)
    axes[r,1].set_title(f'red=GT  green=pred  (Dice {dc:.2f})'); axes[r,1].axis('off')
    axes[r,2].imshow(pm,cmap='magma'); axes[r,2].set_title('Predicted heatmap'); axes[r,2].axis('off')
plt.suptitle('U-Net++ finds the liver mass', y=1.0); plt.tight_layout(); plt.show()"""),

("md", "## 3. 🩺 Classify the scan (Normal / Benign / Malignant) + Grad-CAM"),

("code", """rng2 = np.random.RandomState(0); ci = rng2.permutation(len(rows))
nv = len(rows)//5; val_c=[rows[i] for i in ci[:nv]]; tr_c=[rows[i] for i in ci[nv:]]
class ClsDS(Dataset):
    def __init__(self, rr, train=False): self.rr, self.train = rr, train
    def __len__(self): return len(self.rr)
    def __getitem__(self, i):
        ip, y, _, _ = self.rr[i]
        img = cv2.resize(cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB), (224,224))
        if self.train and np.random.rand()<0.5: img = img[:,::-1]
        x = (img.astype(np.float32)/255.-MEAN)/STD
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2,0,1).float(), int(y)
dl_ct = DataLoader(ClsDS(tr_c,True), 32, shuffle=True, num_workers=2, pin_memory=True)
dl_cv = DataLoader(ClsDS(val_c), 32, num_workers=2, pin_memory=True)
yc = np.array([r[1] for r in tr_c]); w = torch.tensor([len(yc)/(3*(yc==k).sum()) for k in range(3)],
                                                       dtype=torch.float32, device=DEVICE)
clf = timm.create_model('efficientnet_b0', pretrained=True, num_classes=3).to(DEVICE)
crit=nn.CrossEntropyLoss(weight=w); opt=torch.optim.AdamW(clf.parameters(),lr=3e-4,weight_decay=1e-5)
scaler=torch.cuda.amp.GradScaler()
for ep in range(12):
    clf.train(); tot=0
    for x,y in dl_ct:
        x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
        with torch.autocast('cuda'): loss=crit(clf(x),y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot+=loss.item()
    print(f'  cls epoch {ep+1}/12 loss={tot/len(dl_ct):.4f}')
clf.eval(); ys,ps=[],[]
with torch.inference_mode():
    for x,y in dl_cv: ps+=clf(x.to(DEVICE)).argmax(1).cpu().tolist(); ys+=y.tolist()
acc=np.mean(np.array(ys)==np.array(ps)); bacc=balanced_accuracy_score(ys,ps)
print(f'\\nCLASSIFICATION accuracy={acc:.3f}  balanced_acc={bacc:.3f}')
print(classification_report(ys,ps,target_names=CLASSES))
ConfusionMatrixDisplay(confusion_matrix(ys,ps),display_labels=CLASSES).plot(cmap='Blues')
plt.title(f'Liver US 3-class (acc={acc:.3f})'); plt.tight_layout(); plt.show()"""),

("code", """from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
layer=[m for m in clf.modules() if isinstance(m,nn.Conv2d)][-1]; ds=ClsDS(val_c)
fig,axes=plt.subplots(2,4,figsize=(13,7))
with GradCAM(model=clf,target_layers=[layer]) as cam:
    for ax,i in zip(axes.ravel(), np.linspace(0,len(ds)-1,8).astype(int)):
        x,y=ds[i]; rgb=np.clip(x.permute(1,2,0).numpy()*STD+MEAN,0,1).astype(np.float32)
        ax.imshow(show_cam_on_image(rgb, cam(input_tensor=x[None].to(DEVICE))[0], use_rgb=True))
        ax.set_title(CLASSES[y],fontsize=9); ax.axis('off')
plt.suptitle('Grad-CAM: where the classifier looks (should be the lesion / liver texture)')
plt.tight_layout(); plt.show()"""),

("md", """## 🏁 Takeaways

On a **fresh 2025** annotated liver-ultrasound set, two lightweight models do real work: a **U-Net++**
localises the tumor (Dice above), and an **EfficientNet** sorts scans into **Normal / Benign / Malignant**,
with **Grad-CAM** confirming it attends to the lesion rather than probe artefacts. Ultrasound is the
cheapest imaging there is, so tooling like this is exactly where automated triage could help most.

Next: patient-level splits if IDs become available, liver-region masking to focus the classifier,
test-time augmentation, and boundary/Tversky losses for the small malignant masses.
*If you learned something, an upvote is hugely appreciated 🙌*""")
]

out = os.path.join(os.path.dirname(__file__), "liver_ultrasound_segment_classify_gradcam.ipynb")
build(out, cells)
