"""Consolidated IMC experiment harness — sweep matchers / BA / intrinsics refine.

    python experiment.py <scene_dir> <work_dir> \
        [--matchers disk,aliked] [--max_kp 4096] [--max_dim 1280] \
        [--ba] [--refine_pp] [--score_train NAME] [--tag NAME]

Prints one summary line prefixed [RESULT] <tag> ... for easy sweeping.
Sparse keypoint matchers (DISK, ALIKED) share one COLMAP keypoint index space.
"""
import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pycolmap
import torch
import kornia as K
import kornia.feature as KF

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXT = {".png", ".jpg", ".jpeg"}


def load_rgb(path, max_dim):
    img = K.io.load_image(str(path), K.io.ImageLoadType.RGB32, device=DEV)[None]
    h, w = img.shape[-2:]
    s = min(1.0, max_dim / max(h, w))
    if s < 1.0:
        img = K.geometry.resize(img, (round(h * s), round(w * s)), antialias=True)
    return img, s, (h, w)


def _topk(kps, desc, scores, k):
    if kps.shape[0] <= k:
        return kps, desc
    idx = torch.topk(scores, k).indices
    return kps[idx], desc[idx]


@torch.inference_mode()
def extract(paths, matchers, max_dim, max_kp):
    models = {}
    if "disk" in matchers:
        models["disk"] = KF.DISK.from_pretrained("depth").to(DEV).eval()
    if "aliked" in matchers:
        models["aliked"] = KF.ALIKED(model_name="aliked-n16").to(DEV).eval()
    feats = {}
    for p in paths:
        rgb, s, (H, W) = load_rgb(p, max_dim)
        entry = {"hw": (H, W)}
        if "disk" in models:
            df = models["disk"](rgb, max_kp, pad_if_not_divisible=True)[0]
            entry["disk"] = (df.keypoints / s, df.descriptors)
        if "aliked" in models:
            af = models["aliked"](rgb)[0]
            ak, ad = _topk(af.keypoints, af.descriptors, af.keypoint_scores, max_kp)
            entry["aliked"] = (ak / s, ad)
        feats[p.name] = entry
    del models
    torch.cuda.empty_cache()
    return feats


@torch.inference_mode()
def match_all(feats, pairs, matchers, min_matches):
    lg = {k: KF.LightGlueMatcher(k).to(DEV).eval() for k in matchers}
    out = {}
    for a, b in pairs:
        hw_a = torch.tensor(feats[a]["hw"]); hw_b = torch.tensor(feats[b]["hw"])
        off_a = off_b = 0
        combined = []
        for kind in matchers:  # order fixes the global index layout (same as build_db concat)
            ka, da = feats[a][kind]; kb, db = feats[b][kind]
            _, idxs = lg[kind](da, db,
                               KF.laf_from_center_scale_ori(ka[None].float()),
                               KF.laf_from_center_scale_ori(kb[None].float()),
                               hw1=hw_a, hw2=hw_b)
            if len(idxs):
                m = idxs.cpu().numpy().astype(np.int64)
                m[:, 0] += off_a; m[:, 1] += off_b
                combined.append(m)
            off_a += ka.shape[0]; off_b += kb.shape[0]
        if combined:
            m = np.concatenate(combined).astype(np.uint32)
            if len(m) >= min_matches:
                out[(a, b)] = m
    return out


def build_db(feats, matchers, matches, db_path):
    if db_path.exists():
        db_path.unlink()
    db = pycolmap.Database.open(str(db_path))
    ids = {}
    for name, f in feats.items():
        H, W = f["hw"]
        kps = torch.cat([f[k][0] for k in matchers], 0).cpu().numpy().astype(np.float32)
        cam = pycolmap.Camera(model="SIMPLE_RADIAL", width=W, height=H,
                              params=[1.2 * max(H, W), W / 2, H / 2, 0.0])
        cid = db.write_camera(cam, use_camera_id=False)
        iid = db.write_image(pycolmap.Image(name=name, camera_id=cid), use_image_id=False)
        ids[name] = iid
        db.write_keypoints(iid, kps)
    pairs_txt = db_path.parent / "pairs.txt"
    with open(pairs_txt, "w") as fh:
        for (a, b), m in matches.items():
            db.write_matches(ids[a], ids[b], m)
            fh.write(f"{a} {b}\n")
    db.close()
    return pairs_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir"); ap.add_argument("work_dir")
    ap.add_argument("--matchers", default="disk")
    ap.add_argument("--max_kp", type=int, default=4096)
    ap.add_argument("--max_dim", type=int, default=1280)
    ap.add_argument("--min_matches", type=int, default=15)
    ap.add_argument("--ba", action="store_true")
    ap.add_argument("--refine_pp", action="store_true")
    ap.add_argument("--score_train", default=None)
    ap.add_argument("--tag", default="exp")
    args = ap.parse_args()
    matchers = args.matchers.split(",")

    scene = Path(args.scene_dir); work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in scene.iterdir() if p.suffix.lower() in IMG_EXT)
    t_all = time.time()
    feats = extract(paths, matchers, args.max_dim, args.max_kp)
    pairs = list(combinations([p.name for p in paths], 2))
    matches = match_all(feats, pairs, matchers, args.min_matches)

    db_path = work / "database.db"
    pairs_txt = build_db(feats, matchers, matches, db_path)
    pycolmap.verify_matches(str(db_path), str(pairs_txt))

    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = True
    opts.ba_refine_principal_point = bool(args.refine_pp)
    maps = pycolmap.incremental_mapping(str(db_path), str(scene), str(work / "sparse"), opts)

    if args.ba:
        ba_opts = pycolmap.BundleAdjustmentOptions()
        for rec in maps.values():
            pycolmap.bundle_adjustment(rec, ba_opts)

    preds, total = {}, 0
    for rec in maps.values():
        total += rec.num_reg_images()
        for _, img in rec.images.items():
            if img.has_pose:
                pose = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
                preds.setdefault(img.name, (pose.rotation.matrix(), np.asarray(pose.translation)))

    dt = time.time() - t_all
    reg = total / max(len(paths), 1)
    line = f"[RESULT] {args.tag}: reg={reg:.0%} ({total}/{len(paths)}) pairs={len(matches)} time={dt:.0f}s"
    if args.score_train:
        import imc_metric
        root = scene.parents[1]
        res = imc_metric.score_dataset(preds, root / "train_labels.csv",
                                       root / "train_thresholds.csv", args.score_train)
        rot = np.mean([m["rot_mAA"] for m in res.values()])
        tr = np.mean([m["trans_mAA"] for m in res.values()])
        line += f" rot_mAA={rot:.3f} trans_mAA={tr:.3f}"
    print(line)


if __name__ == "__main__":
    main()
