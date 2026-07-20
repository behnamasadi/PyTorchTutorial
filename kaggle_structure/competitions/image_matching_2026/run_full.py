"""Full IMC pipeline over all datasets → submission.csv + train mAA.

Per dataset: DISK features → DINOv2 retrieval pairs (scales to 200+ imgs) →
LightGlue match → pycolmap incremental mapping (COLMAP splits into models = scenes)
→ each model is a predicted scene (clusterN). Composes the validated components.

    python run_full.py --split train --score      # full-train mAA
    python run_full.py --split test                # write submission.csv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap

import experiment as E
import retrieval as R
import rotation as ROT
import imc_metric
import sfm_diagnostics as D

DIAG = {}  # dataset -> post-reconstruction health metrics (why it fails)

IMG_EXT = {".png", ".jpg", ".jpeg"}
DATA = Path("data/extracted")


MATCHERS = ["disk"]  # overridden in main() via --matchers
TUNED_VERIFY = False  # overridden in main() via --tuned_verify
PER_CLUSTER = False   # overridden in main() via --per_cluster
FINAL_BA = False      # overridden in main() via --final_ba (extra global BA → translation)
FIX_ROTATION = False  # overridden in main() via --fix_rotation (re-orient rotated images)
ROT_SIGN = 1          # overridden in main() via --rot_sign (pose un-rotation convention)
MIN_MATCHES = 15      # overridden in main() via --min_matches (lower = more coverage on hard scenes)


SEEDS = 1             # overridden in main() via --seeds (multi-seed best-of-N mapping)
SEED_SELECT = "obs"   # overridden in main() via --seed_select (obs|points|reproj)


def _health_score(maps):
    """Unsupervised reconstruction-quality proxy to pick the best seed on TEST
    (no GT). Primary = total registered images; tie-break by the SEED_SELECT
    health metric (COLMAP mapping is highly stochastic — heritage swings 0.07–0.22
    at the same config, so best-of-N on a good proxy recovers the lucky roll)."""
    reg = pts = obs = 0
    rsum = rn = 0.0
    for rec in maps.values():
        try: reg += rec.num_reg_images()
        except Exception: pass
        try: pts += rec.num_points3D()
        except Exception: pass
        try: obs += rec.compute_num_observations()
        except Exception: pass
        try:
            r = rec.compute_mean_reprojection_error()
            if r: rsum += r; rn += 1
        except Exception: pass
    tie = {"obs": obs, "points": pts,
           "reproj": -(rsum / rn) if rn else -1e9}.get(SEED_SELECT, obs)
    return (reg, tie)


def _map_best(db_path, img_dir, work, opts):
    """Run incremental_mapping SEEDS times; return the model set with the best
    _health_score. SEEDS==1 → single run (baseline behaviour)."""
    if SEEDS <= 1:
        return pycolmap.incremental_mapping(str(db_path), str(img_dir), str(work / "sparse"), opts)
    best_maps, best = None, None
    for si in range(SEEDS):
        m = pycolmap.incremental_mapping(str(db_path), str(img_dir), str(work / f"sparse{si}"), opts)
        sc = _health_score(m)
        if best is None or sc > best:
            best, best_maps = sc, m
    return best_maps


def _final_ba(maps):
    """Extra global bundle adjustment per model — refines camera centers (translation).
    incremental_mapping already BAs, but a final full pass can tighten positions."""
    ba = pycolmap.BundleAdjustmentOptions()
    for rec in maps.values():
        try:
            pycolmap.bundle_adjustment(rec, ba)
        except Exception as e:
            print(f"[full]   final BA skipped: {str(e)[:80]}", flush=True)


def _reconstruct(cluster_names, feats, matches, img_dir, work, scene_tag):
    """Build DB + map for one image group; return {name:{scene,R,t}} for registered."""
    work.mkdir(parents=True, exist_ok=True)
    db_path = work / "database.db"
    cfeats = {n: feats[n] for n in cluster_names}
    pairs_txt = E.build_db(cfeats, MATCHERS, matches, db_path)
    if TUNED_VERIFY:
        tvg = pycolmap.TwoViewGeometryOptions()
        tvg.min_num_inliers = 20; tvg.ransac.max_error = 4.0
        tvg.ransac.confidence = 0.99999; tvg.ransac.max_num_trials = 100000
        pycolmap.verify_matches(str(db_path), str(pairs_txt), tvg)
    else:
        pycolmap.verify_matches(str(db_path), str(pairs_txt))
    opts = pycolmap.IncrementalPipelineOptions(); opts.ba_refine_focal_length = True
    maps = pycolmap.incremental_mapping(str(db_path), str(img_dir), str(work / "sparse"), opts)
    rows = {}
    for midx, rec in maps.items():
        for _, img in rec.images.items():
            if img.has_pose and img.name not in rows:
                pose = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
                rows[img.name] = {"scene": f"{scene_tag}_{midx}",
                                  "R": pose.rotation.matrix(), "t": np.asarray(pose.translation)}
    return rows


def run_dataset(paths, max_kp, max_dim, topk, gt_scene=None):
    names = [p.name for p in paths]
    emb = R.dinov3_descriptors(paths)  # DINOv3: cleaner same-scene retrieval than v2
    img_dir = paths[0].parent
    work_root = Path("work/full") / img_dir.name

    if PER_CLUSTER:
        # DINOv3 clustering DRIVES reconstruction: reconstruct each scene in isolation
        feats = E.extract(paths, MATCHERS, max_dim, max_kp)
        labels = R.cluster_scenes(emb, min_sim=0.5)
        name_arr = np.array(names)
        rows = {}
        for cid in sorted(set(labels)):
            idxs = np.where(labels == cid)[0]
            if len(idxs) < 3:
                continue
            cnames = name_arr[idxs].tolist()
            pairs = R.topk_pairs(emb[idxs], cnames, k=topk)  # pairs WITHIN cluster only
            cmatches = E.match_all({n: feats[n] for n in cnames}, pairs, MATCHERS, min_matches=MIN_MATCHES)
            rows.update(_reconstruct(cnames, feats, cmatches, img_dir,
                                     work_root / f"c{cid}", f"cluster{cid}"))
        return rows

    if FIX_ROTATION:
        kmap = ROT.detect_rotations(paths, emb, matchers=MATCHERS, max_dim=max_dim)
        nrot = sum(1 for v in kmap.values() if v)
        print(f"[full]   rotation-fix: re-oriented {nrot}/{len(names)} images", flush=True)
        feats = ROT.extract_at(paths, kmap, matchers=MATCHERS, max_dim=max_dim, max_kp=max_kp)
    else:
        kmap = {n: 0 for n in names}
        feats = E.extract(paths, MATCHERS, max_dim, max_kp)

    pairs = R.topk_pairs(emb, names, k=topk)
    matches = E.match_all(feats, pairs, MATCHERS, min_matches=MIN_MATCHES)
    if FIX_ROTATION:
        # matches were found in each image's rotated frame; rewrite keypoints back
        # to original coords so COLMAP reconstructs original-image poses directly
        feats = ROT.to_original_frame(feats, kmap, MATCHERS)
    work = Path("work/full") / paths[0].parent.name
    work.mkdir(parents=True, exist_ok=True)
    db_path = work / "database.db"
    pairs_txt = E.build_db(feats, MATCHERS, matches, db_path)
    if TUNED_VERIFY:
        # evidence-based: drop 0-inlier (repeated-structure) pairs + more RANSAC sampling
        # (mitigates the RANSAC stopping-criterion undersampling in few-inlier scenes)
        tvg = pycolmap.TwoViewGeometryOptions()
        tvg.min_num_inliers = 20
        tvg.ransac.max_error = 4.0
        tvg.ransac.confidence = 0.99999
        tvg.ransac.max_num_trials = 100000
        pycolmap.verify_matches(str(db_path), str(pairs_txt), tvg)
    else:
        pycolmap.verify_matches(str(db_path), str(pairs_txt))
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = True
    maps = _map_best(db_path, paths[0].parent, work, opts)
    if FINAL_BA:
        _final_ba(maps)
    # post-reconstruction health (why a dataset is weak, beyond registration rate)
    DIAG[paths[0].parent.name] = D.diagnose(maps, names, gt_scene=gt_scene)
    # each model index = a predicted scene; keep an image's pose from its first model
    rows = {}
    for midx, rec in maps.items():
        for _, img in rec.images.items():
            if img.has_pose and img.name not in rows:
                pose = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
                rows[img.name] = {"scene": f"cluster{midx}",
                                  "R": pose.rotation.matrix(), "t": np.asarray(pose.translation)}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_kp", type=int, default=4096)
    ap.add_argument("--max_dim", type=int, default=1280)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of datasets")
    ap.add_argument("--matchers", default="disk", help="comma: disk,aliked")
    ap.add_argument("--tuned_verify", action="store_true", help="strict RANSAC verification")
    ap.add_argument("--per_cluster", action="store_true", help="DINOv3-cluster then reconstruct per scene")
    ap.add_argument("--final_ba", action="store_true", help="extra global BA after mapping (translation)")
    ap.add_argument("--fix_rotation", action="store_true", help="detect + correct 90/180/270 rotated images (coverage)")
    ap.add_argument("--rot_sign", type=int, default=1, help="pose un-rotation convention (1 or -1); flip if rot_mAA regresses")
    ap.add_argument("--min_matches", type=int, default=15, help="min matches per pair (lower = more coverage on hard scenes)")
    ap.add_argument("--seeds", type=int, default=1, help="multi-seed best-of-N mapping (fights COLMAP variance)")
    ap.add_argument("--seed_select", default="obs", help="seed picker: obs|points|reproj")
    args = ap.parse_args()
    global MATCHERS, TUNED_VERIFY, PER_CLUSTER, FINAL_BA, FIX_ROTATION, ROT_SIGN, MIN_MATCHES, SEEDS, SEED_SELECT
    MATCHERS = args.matchers.split(",")
    TUNED_VERIFY = args.tuned_verify
    PER_CLUSTER = args.per_cluster
    FINAL_BA = args.final_ba
    FIX_ROTATION = args.fix_rotation
    MIN_MATCHES = args.min_matches
    SEEDS = args.seeds
    SEED_SELECT = args.seed_select
    ROT_SIGN = args.rot_sign

    split_dir = DATA / args.split
    datasets = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    if args.only:
        datasets = [d for d in datasets if d in set(args.only.split(","))]

    # GT scene labels (train only) drive per-GT-scene coherence in diagnostics
    gt_by_ds = {}
    labels_csv = DATA / "train_labels.csv"
    if args.split == "train" and labels_csv.exists():
        lab = pd.read_csv(labels_csv)
        for ds_name, grp in lab.groupby("dataset"):
            gt_by_ds[ds_name] = dict(zip(grp["image"], grp["scene"]))

    all_preds = {}
    for ds in datasets:
        paths = sorted(p for p in (split_dir / ds).iterdir() if p.suffix.lower() in IMG_EXT)
        t0 = time.time()
        rows = run_dataset(paths, args.max_kp, args.max_dim, args.topk,
                           gt_scene=gt_by_ds.get(ds))
        reg = len(rows) / max(len(paths), 1)
        all_preds[ds] = rows
        line = f"[full] {ds}: {len(paths)} imgs, reg={reg:.0%}, {time.time()-t0:.0f}s"
        if args.score:
            preds = {n: (v["R"], v["t"]) for n, v in rows.items()}
            res = imc_metric.score_dataset(preds, DATA / "train_labels.csv",
                                           DATA / "train_thresholds.csv", ds)
            if res:
                line += (f"  COMBINED={np.mean([m['combined_mAA'] for m in res.values()]):.3f}"
                         f"  (rot={np.mean([m['rot_mAA'] for m in res.values()]):.3f}"
                         f" trans={np.mean([m['trans_mAA'] for m in res.values()]):.3f})")
        print(line, flush=True)
        if ds in DIAG:
            print(D.evidence_table(DIAG[ds], ds), flush=True)

    # cache poses so submissions / re-scoring don't need re-reconstruction
    import json
    cache = {ds: {n: {"scene": v["scene"], "R": v["R"].tolist(), "t": v["t"].tolist()}
                  for n, v in rows.items()} for ds, rows in all_preds.items()}
    cache_path = Path("work") / f"poses_{args.split}.json"
    cache_path.parent.mkdir(exist_ok=True)
    cache_path.write_text(json.dumps(cache))
    print(f"[full] cached poses -> {cache_path}")

    # cache diagnostics so we can correlate SfM health against mAA offline
    diag_path = Path("work") / f"diag_{args.split}.json"
    diag_path.write_text(json.dumps(DIAG, default=lambda o: None))
    print(f"[full] cached diagnostics -> {diag_path}")

    if args.split == "test":
        sample = pd.read_csv(DATA / "sample_submission.csv")
        for i, row in sample.iterrows():
            p = all_preds.get(row["dataset"], {}).get(row["image"])
            if p:
                sample.at[i, "scene"] = p["scene"]
                sample.at[i, "rotation_matrix"] = ";".join(f"{v:.9f}" for v in p["R"].ravel())
                sample.at[i, "translation_vector"] = ";".join(f"{v:.9f}" for v in p["t"])
        sample.to_csv("submission.csv", index=False)
        print(f"[full] wrote submission.csv ({len(sample)} rows)")

    print("[full] done")


if __name__ == "__main__":
    main()
