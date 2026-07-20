"""Ensemble deep-matching SfM: DISK + ALIKED, both + LightGlue -> pycolmap.

Per image, keypoints from both detectors are concatenated into one COLMAP keypoint
set; each detector matches with its own LightGlue, and ALIKED match indices are
offset by n_disk so both match sets share one index space. This is the standard
IMC "ensemble of sparse matchers" that beats any single detector.

    python run_ensemble.py <scene_dir> <work_dir> [--max_kp 4096] [--max_dim 1280] [--score_train NAME]
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


def load_rgb(path: Path, max_dim: int):
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
def extract(paths, max_dim, max_kp):
    """Return {name: {'disk':(kps,desc), 'aliked':(kps,desc), 'hw':(H,W)}} at orig scale."""
    disk = KF.DISK.from_pretrained("depth").to(DEV).eval()
    aliked = KF.ALIKED(model_name="aliked-n16").to(DEV).eval()
    feats = {}
    for p in paths:
        rgb, s, (H, W) = load_rgb(p, max_dim)
        df = disk(rgb, max_kp, pad_if_not_divisible=True)[0]
        af = aliked(rgb)[0]
        ak, ad = _topk(af.keypoints, af.descriptors, af.keypoint_scores, max_kp)
        feats[p.name] = {
            "disk": (df.keypoints / s, df.descriptors),
            "aliked": (ak / s, ad),
            "hw": (H, W),
        }
    del disk, aliked
    torch.cuda.empty_cache()
    return feats


@torch.inference_mode()
def match_all(feats, pairs, min_matches):
    matchers = {"disk": KF.LightGlueMatcher("disk").to(DEV).eval(),
                "aliked": KF.LightGlueMatcher("aliked").to(DEV).eval()}
    out = {}
    for a, b in pairs:
        hw_a = torch.tensor(feats[a]["hw"]); hw_b = torch.tensor(feats[b]["hw"])
        n_disk_a = feats[a]["disk"][0].shape[0]
        n_disk_b = feats[b]["disk"][0].shape[0]
        combined = []
        for kind, off_a, off_b in (("disk", 0, 0), ("aliked", n_disk_a, n_disk_b)):
            ka, da = feats[a][kind]; kb, db = feats[b][kind]
            lafs_a = KF.laf_from_center_scale_ori(ka[None].float())
            lafs_b = KF.laf_from_center_scale_ori(kb[None].float())
            _, idxs = matchers[kind](da, db, lafs_a, lafs_b, hw1=hw_a, hw2=hw_b)
            if len(idxs):
                m = idxs.cpu().numpy().astype(np.int64)
                m[:, 0] += off_a; m[:, 1] += off_b
                combined.append(m)
        if combined:
            m = np.concatenate(combined).astype(np.uint32)
            if len(m) >= min_matches:
                out[(a, b)] = m
    return out


def build_db(feats, matches, db_path: Path):
    if db_path.exists():
        db_path.unlink()
    db = pycolmap.Database.open(str(db_path))
    ids = {}
    for name, f in feats.items():
        H, W = f["hw"]
        kps = torch.cat([f["disk"][0], f["aliked"][0]], 0).cpu().numpy().astype(np.float32)
        cam = pycolmap.Camera(model="SIMPLE_RADIAL", width=W, height=H,
                              params=[1.2 * max(H, W), W / 2, H / 2, 0.0])
        cid = db.write_camera(cam, use_camera_id=False)
        img = pycolmap.Image(name=name, camera_id=cid)
        iid = db.write_image(img, use_image_id=False)
        ids[name] = iid
        db.write_keypoints(iid, kps)
    pairs_txt = db_path.parent / "pairs.txt"
    with open(pairs_txt, "w") as fh:
        for (a, b), m in matches.items():
            db.write_matches(ids[a], ids[b], m)
            fh.write(f"{a} {b}\n")
    db.close()
    return pairs_txt


def collect_and_score(maps, paths, scene, args):
    preds, total = {}, 0
    for idx, rec in maps.items():
        n = rec.num_reg_images(); total += n
        print(f"  model {idx}: registered {n}/{len(paths)}")
        for _, img in rec.images.items():
            if not img.has_pose:
                continue
            pose = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
            preds.setdefault(img.name, (pose.rotation.matrix(), np.asarray(pose.translation)))
    print(f"[ens] TOTAL registered {total}/{len(paths)} ({total/max(len(paths),1):.0%})")
    if args.score_train:
        import imc_metric
        root = scene.parents[1]
        res = imc_metric.score_dataset(preds, root / "train_labels.csv",
                                       root / "train_thresholds.csv", args.score_train)
        for sc, m in res.items():
            print(f"  {sc:24s} reg {m['registered']:>3}/{m['n']:<3} "
                  f"rot_mAA={m['rot_mAA']:.3f} trans_mAA={m['trans_mAA']:.3f}")
        print(f"[score] MEAN trans_mAA={np.mean([m['trans_mAA'] for m in res.values()]):.3f} "
              f"rot_mAA={np.mean([m['rot_mAA'] for m in res.values()]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir"); ap.add_argument("work_dir")
    ap.add_argument("--max_kp", type=int, default=4096)
    ap.add_argument("--max_dim", type=int, default=1280)
    ap.add_argument("--min_matches", type=int, default=15)
    ap.add_argument("--score_train", default=None)
    args = ap.parse_args()

    scene = Path(args.scene_dir); work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in scene.iterdir() if p.suffix.lower() in IMG_EXT)
    print(f"[ens] scene={scene.name} images={len(paths)} device={DEV}")

    t0 = time.time(); feats = extract(paths, args.max_dim, args.max_kp)
    print(f"[ens] DISK+ALIKED features ({time.time()-t0:.1f}s)")
    pairs = list(combinations([p.name for p in paths], 2))
    t0 = time.time(); matches = match_all(feats, pairs, args.min_matches)
    print(f"[ens] matched {len(matches)}/{len(pairs)} pairs ({time.time()-t0:.1f}s)")

    db_path = work / "database.db"
    pairs_txt = build_db(feats, matches, db_path)
    pycolmap.verify_matches(str(db_path), str(pairs_txt))
    t0 = time.time(); maps = pycolmap.incremental_mapping(str(db_path), str(scene), str(work / "sparse"))
    print(f"[ens] mapping ({time.time()-t0:.1f}s) models={len(maps)}")
    collect_and_score(maps, paths, scene, args)


if __name__ == "__main__":
    main()
