"""DINOv2 global descriptors → scene clustering + retrieval pairing.

Enables scaling beyond exhaustive matching: the big IMC datasets have 200+ images
(20k+ exhaustive pairs). We (1) cluster images into scenes by descriptor similarity
(datasets mix multiple scenes + outliers) and (2) within each scene generate only
top-k retrieval pairs to match.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import kornia as K


DINOV3_PATH = "imc_deps/dinov3_vitb16"  # local HF snapshot (gated weights)


@torch.inference_mode()
def dinov3_descriptors(paths, size=224, device="cuda"):
    """(N, D) L2-normalized DINOv3 ViT-B/16 CLS descriptors — cleaner scene
    clustering than v2 (splits similar landmarks that v2 merged)."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(DINOV3_PATH).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    for p in paths:
        img = K.io.load_image(str(p), K.io.ImageLoadType.RGB32, device=device)[None]
        img = (K.geometry.resize(img, (size, size), antialias=True) - mean) / std
        out = model(pixel_values=img)
        cls = out.pooler_output if getattr(out, "pooler_output", None) is not None \
            else out.last_hidden_state[:, 0]
        feats.append(cls.squeeze(0).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    emb = np.stack(feats)
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)


@torch.inference_mode()
def dinov2_descriptors(paths, model_name="dinov2_vits14", size=518, device="cuda"):
    """(N, D) L2-normalized DINOv2 CLS descriptors (fallback / open-weights)."""
    model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    for p in paths:
        img = K.io.load_image(str(p), K.io.ImageLoadType.RGB32, device=device)[None]
        img = K.geometry.resize(img, (size, size), antialias=True)
        img = (img - mean) / std
        feats.append(model(img).squeeze(0).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    emb = np.stack(feats)
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)


def cluster_scenes(emb, min_sim=0.5):
    """Agglomerative clustering on cosine distance → scene id per image."""
    from sklearn.cluster import AgglomerativeClustering

    n = len(emb)
    if n <= 2:
        return np.zeros(n, dtype=int)
    dist = np.clip(1.0 - emb @ emb.T, 0, 2)
    np.fill_diagonal(dist, 0.0)
    return AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average",
        distance_threshold=1.0 - min_sim,
    ).fit_predict(dist)


def topk_pairs(emb, names, k=30, exhaustive_max=50):
    """Retrieval pairs (name_a, name_b); exhaustive if few images."""
    n = len(names)
    if n <= exhaustive_max:
        return [(names[i], names[j]) for i in range(n) for j in range(i + 1, n)]
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1)
    pairs = set()
    for i in range(n):
        for j in np.argsort(-sim[i])[:k]:
            a, b = names[i], names[int(j)]
            pairs.add((a, b) if a < b else (b, a))
    return sorted(pairs)


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1])
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    emb = dinov2_descriptors(paths)
    labels = cluster_scenes(emb)
    names = [p.name for p in paths]
    pairs = topk_pairs(emb, names, k=30)
    print(f"images={len(paths)} clusters={len(set(labels))} "
          f"cluster_sizes={np.bincount(labels).tolist()} pairs={len(pairs)} "
          f"(exhaustive would be {len(paths)*(len(paths)-1)//2})")
