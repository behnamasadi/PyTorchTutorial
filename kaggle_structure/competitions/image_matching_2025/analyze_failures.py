"""Diagnose WHY images fail to register on the weak datasets.

For a dataset: extract DISK, DINOv3 retrieval pairs, LightGlue match, RANSAC verify,
then map. Reports metrics to localize the failure mode:
  - match-count distribution (matching quality)
  - #pairs above thresholds (graph density)
  - RANSAC inlier ratios (geometry quality)
  - match-graph connected components + isolated images (pairing/retrieval)
  - registration vs #good-pairs-per-image correlation (mapper)

    python analyze_failures.py <dataset_name> [--topk 30]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycolmap

import experiment as E
import retrieval as R

DATA = Path("data/extracted")
IMG_EXT = {".png", ".jpg", ".jpeg"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--max_kp", type=int, default=4096)
    ap.add_argument("--max_dim", type=int, default=1280)
    args = ap.parse_args()
    E.MATCHERS = ["disk"]

    ds_dir = DATA / "train" / args.dataset
    paths = sorted(p for p in ds_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    names = [p.name for p in paths]
    n = len(paths)
    print(f"\n===== {args.dataset}: {n} images =====", flush=True)

    feats = E.extract(paths, ["disk"], args.max_dim, args.max_kp)
    emb = R.dinov3_descriptors(paths)
    pairs = R.topk_pairs(emb, names, k=args.topk)
    matches = E.match_all(feats, pairs, ["disk"], min_matches=1)  # keep all to see distribution

    counts = np.array([len(m) for m in matches.values()])
    print(f"[match] candidate pairs={len(pairs)}  matched-pairs(>=1)={len(matches)}")
    if len(counts):
        print(f"[match] matches/pair: min={counts.min()} median={int(np.median(counts))} "
              f"max={counts.max()} mean={counts.mean():.0f}")
        for thr in (15, 50, 100, 300):
            print(f"[match]   pairs with >={thr} matches: {(counts >= thr).sum()}")

    # match graph (edges = pairs with >=15 matches)
    deg = defaultdict(int)
    good_edges = [(a, b) for (a, b), m in matches.items() if len(m) >= 15]
    for a, b in good_edges:
        deg[a] += 1; deg[b] += 1
    isolated = [nm for nm in names if deg[nm] == 0]
    print(f"[graph] good edges(>=15)={len(good_edges)}  isolated images={len(isolated)}/{n}")
    degs = np.array([deg[nm] for nm in names])
    print(f"[graph] per-image good-neighbors: min={degs.min()} median={int(np.median(degs))} max={degs.max()}")

    # RANSAC verification: inlier ratios
    work = Path("work/analyze") / args.dataset
    work.mkdir(parents=True, exist_ok=True)
    good_matches = {k: v for k, v in matches.items() if len(v) >= 15}
    pairs_txt = E.build_db(feats, ["disk"], good_matches, work / "database.db")
    db_path = work / "database.db"
    pycolmap.verify_matches(str(db_path), str(pairs_txt))
    db = pycolmap.Database.open(str(db_path))
    inl_ratios = []
    for (a, b) in good_matches:
        try:
            id1 = db.read_image_with_name(a).image_id
            id2 = db.read_image_with_name(b).image_id
            tvg = db.read_two_view_geometry(id1, id2)
            n_inl = len(tvg.inlier_matches)
            inl_ratios.append(n_inl / max(len(good_matches[(a, b)]), 1))
        except Exception:
            pass
    db.close()
    if inl_ratios:
        ir = np.array(inl_ratios)
        print(f"[ransac] verified pairs={len(ir)}  inlier-ratio: median={np.median(ir):.2f} "
              f"mean={ir.mean():.2f}  pairs>0.5={((ir > 0.5).sum())}")

    # mapping: who registers
    opts = pycolmap.IncrementalPipelineOptions(); opts.ba_refine_focal_length = True
    maps = pycolmap.incremental_mapping(str(db_path), str(ds_dir), str(work / "sparse"), opts)
    reg_names = set()
    for rec in maps.values():
        for _, img in rec.images.items():
            if img.has_pose:
                reg_names.add(img.name)
    print(f"[map] models={len(maps)} registered={len(reg_names)}/{n} ({len(reg_names)/n:.0%})")
    # correlation: do registered images have more good-neighbors?
    reg_deg = [deg[nm] for nm in names if nm in reg_names]
    unreg_deg = [deg[nm] for nm in names if nm not in reg_names]
    if reg_deg and unreg_deg:
        print(f"[map] avg good-neighbors: registered={np.mean(reg_deg):.1f} vs "
              f"unregistered={np.mean(unreg_deg):.1f}")


if __name__ == "__main__":
    main()
