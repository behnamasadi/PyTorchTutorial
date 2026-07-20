"""Image Matching Challenge 2026-ongoing — SfM pipeline.

Per dataset:  cluster images into scenes -> match -> RANSAC -> pycolmap -> 6-DoF poses.

Data layout (verified 2026-07-18):
    data/extracted/
        train/<dataset>/<image>.png        # scenes MIXED within a dataset folder
        test/<dataset>/<image>.png
        train_labels.csv                    # dataset,scene,image,rotation_matrix,translation_vector
        train_thresholds.csv                # per-scene mAA thresholds
        sample_submission.csv               # image_id,dataset,scene,image,rotation_matrix,translation_vector

Submission encoding: rotation_matrix = 9 row-major floats joined by ';'; translation_vector = 3 by ';'.

I/O + orchestration + local mAA scoring are concrete. The matcher/SfM internals
(`match_pair`, `reconstruct`) need kornia + pycolmap — present in the Kaggle GPU docker.

Local dev:
    python pipeline.py --config config.yaml --split train --score
Offline notebook: import `run_dataset` + `write_submission`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

IMG_EXT = {".png", ".jpg", ".jpeg"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def list_datasets(split_dir: Path) -> dict[str, list[Path]]:
    """{dataset_name: [image paths]} for every dataset folder under a split."""
    out = {}
    for ds in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        imgs = sorted(p for p in ds.iterdir() if p.suffix.lower() in IMG_EXT)
        if imgs:
            out[ds.name] = imgs
    return out


# --------------------------------------------------------------------------- #
# 1. Retrieval / scene clustering
# --------------------------------------------------------------------------- #
def global_descriptors(image_paths: list[Path], model_name: str) -> np.ndarray:
    """L2-normalized global descriptors (DINOv2) for clustering + pair gen."""
    import torch
    import torchvision.transforms as T
    from PIL import Image

    model = torch.hub.load("facebookresearch/dinov2", model_name).eval().cuda()
    tf = T.Compose([T.Resize(518), T.CenterCrop(518), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    feats = []
    with torch.inference_mode():
        for p in image_paths:
            x = tf(Image.open(p).convert("RGB")).unsqueeze(0).cuda()
            feats.append(model(x).squeeze(0).float().cpu().numpy())
    emb = np.stack(feats)
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)


def cluster_scenes(emb: np.ndarray, min_sim: float = 0.5) -> np.ndarray:
    """Agglomerative clustering on descriptor similarity -> scene id per image.

    A reasonable first pass; the strong version clusters on *verified match counts*
    (covisibility) after matching, then re-runs SfM per cluster.
    """
    from sklearn.cluster import AgglomerativeClustering

    n = len(emb)
    if n <= 2:
        return np.zeros(n, dtype=int)
    dist = 1.0 - (emb @ emb.T)
    np.fill_diagonal(dist, 0.0)
    labels = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average",
        distance_threshold=1.0 - min_sim,
    ).fit_predict(dist)
    return labels


def build_pairs(emb: np.ndarray, idxs: np.ndarray, top_k: int, exhaustive_max: int):
    """Candidate pairs within one scene cluster (indices into the dataset arrays)."""
    if len(idxs) <= exhaustive_max:
        return [(int(idxs[i]), int(idxs[j]))
                for i in range(len(idxs)) for j in range(i + 1, len(idxs))]
    sub = emb[idxs]
    sim = sub @ sub.T
    np.fill_diagonal(sim, -1)
    pairs = set()
    for a in range(len(idxs)):
        for b in np.argsort(-sim[a])[:top_k]:
            i, j = int(idxs[a]), int(idxs[b])
            pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


# --------------------------------------------------------------------------- #
# 2. Matching + geometric verification   (needs kornia — Kaggle docker)
# --------------------------------------------------------------------------- #
def match_pair(path_i: Path, path_j: Path, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble matcher -> (mkpts0, mkpts1). Concatenate all matchers before RANSAC."""
    import kornia.feature as KF  # noqa: F401  (used once implemented)

    all0, all1 = [], []
    for m in cfg["matching"]["matchers"]:
        # TODO(docker): ALIKED/SuperPoint + KF.LightGlueMatcher, or KF.LoFTR('outdoor').
        # Append matched keypoints to all0/all1. Left explicit — needs GPU + weights.
        _ = m
    if not all0:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.concatenate(all0), np.concatenate(all1)


def geometric_verify(mkpts0, mkpts1, cfg):
    import cv2

    if len(mkpts0) < cfg["matching"]["min_matches"]:
        return mkpts0[:0], mkpts1[:0]
    _F, mask = cv2.findFundamentalMat(
        mkpts0, mkpts1, cv2.USAC_MAGSAC,
        cfg["ransac"]["reproj_threshold"], cfg["ransac"]["confidence"],
        cfg["ransac"]["max_iters"])
    if mask is None:
        return mkpts0[:0], mkpts1[:0]
    keep = mask.ravel().astype(bool)
    return mkpts0[keep], mkpts1[keep]


# --------------------------------------------------------------------------- #
# 3. Incremental SfM -> poses            (needs pycolmap — Kaggle docker)
# --------------------------------------------------------------------------- #
def reconstruct(paths: list[Path], pairs, matches, work_dir: Path, cfg) -> dict:
    """Import keypoints+matches into a pycolmap DB, run incremental_mapping.

    Returns {image_name: (R 3x3 np, t 3 np)} for registered images.
    """
    import pycolmap  # noqa: F401

    work_dir.mkdir(parents=True, exist_ok=True)
    # TODO(docker): build pycolmap.Database, import_matches for `pairs`, verify,
    #               incremental_mapping(); read camera poses from the largest model.
    return {}


# --------------------------------------------------------------------------- #
# Orchestration over one dataset
# --------------------------------------------------------------------------- #
def run_dataset(name: str, paths: list[Path], cfg) -> list[dict]:
    """Returns rows: {image, scene(clusterN), R(3x3), t(3)} for one dataset."""
    emb = global_descriptors(paths, cfg["retrieval"]["model"])
    scene_ids = cluster_scenes(emb, cfg["retrieval"].get("cluster_min_sim", 0.5))
    work_root = Path(cfg.get("work_dir", "./work")) / name
    rows = []
    for sid in np.unique(scene_ids):
        idxs = np.where(scene_ids == sid)[0]
        pairs = build_pairs(emb, idxs, cfg["retrieval"]["top_k"],
                            cfg["retrieval"]["exhaustive_max_images"])
        matches = {}
        for i, j in pairs:
            m0, m1 = geometric_verify(*match_pair(paths[i], paths[j], cfg), cfg)
            if len(m0):
                matches[(i, j)] = (m0, m1)
        poses = reconstruct([paths[k] for k in idxs], pairs, matches,
                            work_root / f"cluster{sid}", cfg)
        for k in idxs:
            R, t = poses.get(paths[k].name, (np.eye(3), np.zeros(3)))
            rows.append({"image": paths[k].name, "scene": f"cluster{sid}",
                         "R": R, "t": t})
    return rows


# --------------------------------------------------------------------------- #
# Submission + scoring
# --------------------------------------------------------------------------- #
def _fmt(mat: np.ndarray) -> str:
    return ";".join(f"{v:.9f}" for v in np.asarray(mat).ravel())


def write_submission(sample_csv: Path, preds: dict[str, dict], out_csv: Path):
    """Fill the sample_submission rows with predicted scene + R + t.

    preds: {dataset: {image_name: {"scene": str, "R": 3x3, "t": 3}}}
    """
    df = pd.read_csv(sample_csv)
    for i, row in df.iterrows():
        p = preds.get(row["dataset"], {}).get(row["image"])
        if p is not None:
            df.at[i, "scene"] = p["scene"]
            df.at[i, "rotation_matrix"] = _fmt(p["R"])
            df.at[i, "translation_vector"] = _fmt(p["t"])
    df.to_csv(out_csv, index=False)
    print(f"[imc] wrote {out_csv}  ({len(df)} rows)")


def score_local(preds: dict, labels_csv: Path, thresholds_csv: Path) -> float:
    """Approximate local mAA against train_labels (registration-rate proxy).

    Full metric solves the best scene<->cluster assignment + per-threshold pose
    accuracy; this reports the fraction of images registered as a fast dev signal.
    """
    labels = pd.read_csv(labels_csv)
    total = len(labels)
    registered = sum(
        1 for _, r in labels.iterrows()
        if r["image"] in preds.get(r["dataset"], {})
        and not np.allclose(preds[r["dataset"]][r["image"]]["t"], 0)
    )
    rate = registered / max(total, 1)
    print(f"[imc] registered {registered}/{total} ({rate:.1%}) — proxy, not full mAA")
    return rate


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    root = Path(cfg["data_dir"]) / "extracted"
    split_dir = root / args.split
    datasets = list_datasets(split_dir)
    print(f"[imc] {args.split}: {len(datasets)} datasets — {sum(len(v) for v in datasets.values())} images")

    preds = {}
    for name, paths in datasets.items():
        print(f"[imc] {name}: {len(paths)} images")
        rows = run_dataset(name, paths, cfg)
        preds[name] = {r["image"]: r for r in rows}

    write_submission(root / "sample_submission.csv", preds, Path(cfg["submission"]["out_csv"]))
    if args.score and args.split == "train":
        score_local(preds, root / "train_labels.csv", root / "train_thresholds.csv")


if __name__ == "__main__":
    main()
