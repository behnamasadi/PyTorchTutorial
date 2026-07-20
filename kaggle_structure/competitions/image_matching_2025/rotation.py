"""Rotation correction — the coverage lever for rotated (90/180/270) images.

DISK / ALIKED / LightGlue are NOT rotation-invariant, so an image stored rotated
relative to its neighbours silently fails to match and never registers (our
heritage 47% / vineyard 64% coverage holes). This module estimates a per-image
90-degree rotation label so every image in a scene is matched in ONE consistent
orientation frame, then hands back the chosen-orientation features + a pose
un-rotation so the reported pose is for the ORIGINAL image.

Why relative-only is enough: `imc_metric.score_dataset` Sim3-aligns predicted
camera centres to GT (a global per-scene rotation is absorbed by `R_align`) and
camera centres are invariant to in-plane rotation. So the win is purely more
images REGISTERING; we only need each scene mutually consistent, plus a per-image
`Rz(-theta)` on the reported rotation matrix (centres/translation unaffected).

Estimation is weight-free (no extra model, no internet — Kaggle-offline safe):
for each retrieval pair we try the 4 rotations of one image, vote by LightGlue
match count, and propagate absolute labels over the pair graph by BFS.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import kornia.feature as KF

import experiment as E

DEV = E.DEV


def Rz(k: int) -> np.ndarray:
    """3x3 rotation about the optical (z) axis by k*90 degrees (CCW)."""
    th = np.deg2rad(90.0 * (k % 4))
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float64)


def unrotate_pose(R, t, k: int, sign: int = 1):
    """Map a rotated-image pose back to the ORIGINAL image orientation.

    If image pixels were rotated by k*90 CCW, the camera is rotated about its
    optical axis, so R_orig = Rz(-k)@R_rot, t_orig = Rz(-k)@t_rot. `sign` flips
    the convention (torch.rot90 vs COLMAP handedness) — validated end-to-end by
    the local scorer; flip if rot_mAA regresses on newly-registered images.
    """
    if k % 4 == 0:
        return np.asarray(R, np.float64), np.asarray(t, np.float64)
    Rc = Rz(-sign * k)
    return Rc @ np.asarray(R, np.float64), Rc @ np.asarray(t, np.float64)


def unrotate_keypoints(kp, k, hw_rot):
    """Map keypoints from a k*90-CCW-rotated frame back to ORIGINAL image coords.

    kp: (N,2) [x,y] in rotated coords; hw_rot=(Hr,Wr) rotated dims. Returns
    (kp_orig (N,2), (H,W) original dims). Formulas verified against np.rot90:
      k=1: x=(Hr-1)-yr, y=xr    k=2: x=(Wr-1)-xr, y=(Hr-1)-yr    k=3: x=yr, y=(Wr-1)-xr
    Matching runs in the rotated frame (descriptors are rotation-variant), then we
    rewrite the matched keypoints here so COLMAP reconstructs the ORIGINAL camera
    pose directly — no error-prone pose un-rotation.
    """
    k = k % 4
    Hr, Wr = hw_rot
    if k == 0:
        return kp, (Hr, Wr)
    xr, yr = kp[:, 0], kp[:, 1]
    if k == 1:
        x, y, hw = (Hr - 1) - yr, xr, (Wr, Hr)
    elif k == 2:
        x, y, hw = (Wr - 1) - xr, (Hr - 1) - yr, (Hr, Wr)
    else:  # k == 3
        x, y, hw = yr, (Wr - 1) - xr, (Wr, Hr)
    return torch.stack([x, y], dim=1), hw


def to_original_frame(feats, kmap, matchers=("disk",)):
    """Rewrite matched features from each image's rotated frame back to original
    coords (in place). Match indices are unchanged, so matches stay valid."""
    for name, entry in feats.items():
        k = kmap.get(name, 0) % 4
        if k == 0:
            continue
        hw_rot = entry["hw"]
        new_hw = hw_rot
        for kind in matchers:
            kp, desc = entry[kind]
            kp2, new_hw = unrotate_keypoints(kp, k, hw_rot)
            entry[kind] = (kp2, desc)
        entry["hw"] = new_hw
    return feats


def load_models(matchers):
    models = {}
    if "disk" in matchers:
        models["disk"] = KF.DISK.from_pretrained("depth").to(DEV).eval()
    if "aliked" in matchers:
        models["aliked"] = KF.ALIKED(model_name="aliked-n16").to(DEV).eval()
    return models


@torch.inference_mode()
def _extract_one(path, k, models, max_dim, max_kp, to_cpu=True):
    """Feature entry for `path` rotated by k*90 CCW. Keypoints in rotated
    full-res coords, hw = rotated full-res dims (W,H for odd k)."""
    rgb, s, (H, W) = E.load_rgb(path, max_dim)
    if k % 4:
        rgb = torch.rot90(rgb, k, dims=(-2, -1))
    Hr, Wr = (H, W) if k % 2 == 0 else (W, H)
    entry = {"hw": (Hr, Wr)}
    if "disk" in models:
        df = models["disk"](rgb, max_kp, pad_if_not_divisible=True)[0]
        kp, de = df.keypoints / s, df.descriptors
        entry["disk"] = (kp.cpu(), de.cpu()) if to_cpu else (kp, de)
    if "aliked" in models:
        af = models["aliked"](rgb)[0]
        ak, ad = E._topk(af.keypoints, af.descriptors, af.keypoint_scores, max_kp)
        ak = ak / s
        entry["aliked"] = (ak.cpu(), ad.cpu()) if to_cpu else (ak, ad)
    return entry


def _to_dev(entry, matchers):
    out = {"hw": entry["hw"]}
    for k in matchers:
        kp, de = entry[k]
        out[k] = (kp.to(DEV), de.to(DEV))
    return out


@torch.inference_mode()
def _geo_inliers(lg, ea, eb, matchers) -> int:
    """RANSAC fundamental-matrix inliers between two entries (on DEV).

    Raw LightGlue match count is fooled by repetitive/symmetric texture (a wrong
    rotation still returns many matches), which made match-count voting assign
    bogus rotations. Geometric inliers survive only for the true orientation.
    """
    hw_a = torch.tensor(ea["hw"]); hw_b = torch.tensor(eb["hw"])
    pa, pb = [], []
    for kind in matchers:
        ka, da = ea[kind]; kb, db = eb[kind]
        _, idxs = lg[kind](da, db,
                           KF.laf_from_center_scale_ori(ka[None].float()),
                           KF.laf_from_center_scale_ori(kb[None].float()),
                           hw1=hw_a, hw2=hw_b)
        if len(idxs):
            ii = idxs.cpu().numpy()
            pa.append(ka.cpu().numpy()[ii[:, 0]])
            pb.append(kb.cpu().numpy()[ii[:, 1]])
    if not pa:
        return 0
    pa = np.ascontiguousarray(np.concatenate(pa), dtype=np.float32)
    pb = np.ascontiguousarray(np.concatenate(pb), dtype=np.float32)
    if len(pa) < 8:
        return 0
    try:
        _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.5, 0.999)
    except cv2.error:
        return 0
    return int(mask.sum()) if mask is not None else 0


def _neighbors(emb, k):
    """Per-index top-k neighbour indices from cosine-sim embeddings."""
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1.0)
    return {i: np.argsort(-sim[i])[:k].tolist() for i in range(len(emb))}


@torch.inference_mode()
def detect_rotations(paths, emb, matchers=("disk",), max_dim=1280,
                     det_kp=2048, det_topk=6, min_inliers=25, margin=2.0):
    """Estimate a 90-degree rotation label per image (name -> k in 0..3).

    Zero extra weights: 4-rotation feature extraction + LightGlue match-count
    voting over DINOv3 retrieval neighbours, absolute labels by BFS on the graph.
    """
    matchers = list(matchers)
    names = [p.name for p in paths]
    n = len(names)
    if n < 3:
        return {nm: 0 for nm in names}

    models = load_models(matchers)
    lg = {k: KF.LightGlueMatcher(k).to(DEV).eval() for k in matchers}

    # 4-rotation features (CPU-resident to bound VRAM), det_kp for speed.
    feats_rot = [[None] * n for _ in range(4)]
    for i, p in enumerate(paths):
        for k in range(4):
            feats_rot[k][i] = _extract_one(p, k, models, max_dim, det_kp)

    nbrs = _neighbors(emb, det_topk)
    # canonical undirected edges (i<j); delta measured with i at rot 0.
    seen = set()
    edges = []  # (i, j, delta, weight)
    for i in range(n):
        for j in nbrs[i]:
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            ea0 = _to_dev(feats_rot[0][a], matchers)
            counts = []
            for r in range(4):
                eb = _to_dev(feats_rot[r][b], matchers)
                counts.append(_geo_inliers(lg, ea0, eb, matchers))
            order = np.argsort(counts)[::-1]
            best, second = int(order[0]), int(order[1])
            # accept only confident, geometrically well-separated votes
            if counts[best] >= min_inliers and counts[best] >= margin * max(counts[second], 1):
                edges.append((a, b, best, counts[best]))

    del models, lg, feats_rot
    torch.cuda.empty_cache()

    # BFS over the rotation graph → relative labels per connected component.
    adj = {i: [] for i in range(n)}
    for a, b, delta, w in edges:
        adj[a].append((b, delta, w))       # k_b = k_a + delta
        adj[b].append((a, (-delta) % 4, w))  # k_a = k_b - delta
    label = {}
    comps = []
    for seed in range(n):
        if seed in label or not adj[seed]:
            continue
        label[seed] = 0
        comp = [seed]
        stack = [seed]
        while stack:
            u = stack.pop()
            for v, d, _w in sorted(adj[u], key=lambda e: -e[2]):
                if v not in label:
                    label[v] = (label[u] + d) % 4
                    comp.append(v)
                    stack.append(v)
        comps.append(comp)
    # Normalise each component so its MAJORITY orientation = 0. Only relative
    # labels matter for matching (a per-component global offset is absorbed by the
    # scorer's Sim3), so anchoring on the modal label leaves genuinely-upright
    # images untouched and minimises how many poses need un-rotation.
    for comp in comps:
        vals = [label[i] for i in comp]
        mode = max(set(vals), key=vals.count)
        for i in comp:
            label[i] = (label[i] - mode) % 4
    kmap = {names[i]: int(label.get(i, 0)) for i in range(n)}
    return kmap


@torch.inference_mode()
def extract_at(paths, kmap, matchers=("disk",), max_dim=1280, max_kp=8192):
    """Full-res features at each image's chosen rotation (on DEV, for matching)."""
    matchers = list(matchers)
    models = load_models(matchers)
    feats = {}
    for p in paths:
        feats[p.name] = _extract_one(p, kmap.get(p.name, 0), models,
                                     max_dim, max_kp, to_cpu=False)
    del models
    torch.cuda.empty_cache()
    return feats
