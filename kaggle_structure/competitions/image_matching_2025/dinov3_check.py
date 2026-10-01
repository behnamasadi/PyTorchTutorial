import torch, numpy as np, kornia as K
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoModel
dev='cuda'
def emb_dinov3(paths):
    m=AutoModel.from_pretrained("imc_deps/dinov3_vitb16").to(dev).eval()
    mean=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
    E=[]
    with torch.inference_mode():
        for p in paths:
            img=K.io.load_image(str(p),K.io.ImageLoadType.RGB32,device=dev)[None]
            img=K.geometry.resize(img,(224,224),antialias=True); img=(img-mean)/std
            out=m(pixel_values=img)
            cls=out.pooler_output if getattr(out,'pooler_output',None) is not None else out.last_hidden_state[:,0]
            E.append(cls.squeeze(0).float().cpu().numpy())
    E=np.stack(E); return E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-9)
def clu(emb,ms):
    d=np.clip(1-emb@emb.T,0,2); np.fill_diagonal(d,0)
    return AgglomerativeClustering(n_clusters=None,metric='precomputed',linkage='average',distance_threshold=1-ms).fit_predict(d)
for ds,gt in [("pt_brandenburg_british_buckingham","3x75"),("imc2023_haiper","bike15/chairs16/fountain23")]:
    paths=sorted(p for p in Path(f"data/extracted/train/{ds}").iterdir() if p.suffix.lower()=='.png')
    E=emb_dinov3(paths)
    print(f"\n{ds} (GT {gt}), {len(paths)} imgs:")
    for ms in (0.4,0.5,0.6):
        lb=clu(E,ms); print(f"  DINOv3 min_sim={ms}: {len(set(lb))} clusters {sorted(np.bincount(lb).tolist(),reverse=True)}")
