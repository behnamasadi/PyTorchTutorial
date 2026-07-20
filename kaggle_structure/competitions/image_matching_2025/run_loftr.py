"""LoFTR (detector-free / dense) SfM for hard low-texture scenes.

LoFTR produces per-pair correspondences with no repeatable keypoints, so we
aggregate: quantize each image's matched points to an integer-pixel grid to build
one keypoint set per image, then matches reference those indices. This is the
hloc `match_dense` trick. Intended for the scenes where DISK/ALIKED stall (stairs).

    python run_loftr.py <scene_dir> <work_dir> [--max_dim 840] [--score_train NAME]
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


def load_gray(path, max_dim):
    img = K.io.load_image(str(path), K.io.ImageLoadType.GRAY32, device=DEV)[None]
    h, w = img.shape[-2:]
    s = min(1.0, max_dim / max(h, w))
    nh, nw = round(h * s / 8) * 8, round(w * s / 8) * 8  # LoFTR needs /8
    img = K.geometry.resize(img, (nh, nw), antialias=True)
    return img, (w / nw, h / nh), (h, w)  # per-axis scale back to original


class KeypointAggregator:
    """Merge per-pair dense-matcher points into one keypoint set per image.

    ``merge_radius=0`` → original 1px integer-grid dedup. ``merge_radius=R>0`` →
    snap a new point to an existing keypoint within R px (looked up via an R-sized
    bucket grid + its 8 neighbours). Dense matchers (RoMa/LoFTR) don't produce
    repeatable keypoints across pairs, so the same physical point lands in
    different 1px cells and COLMAP can't form tracks → bad triangulation (the
    ETs trans=0.018 collapse). Radius-merge gives cross-pair track consistency
    while keeping the sub-pixel stored coordinate.
    """
    def __init__(self, merge_radius=0.0):
        self.r = float(merge_radius)
        self.index = {}   # name -> {cell: idx (r=0) | [idx,...] (r>0)}
        self.pts = {}     # name -> list[[x,y]]

    def add(self, name, x, y):
        pts = self.pts.setdefault(name, [])
        d = self.index.setdefault(name, {})
        if self.r <= 0:
            key = (int(round(x)), int(round(y)))
            if key not in d:
                d[key] = len(pts); pts.append([float(x), float(y)])
            return d[key]
        cx, cy = int(x // self.r), int(y // self.r)
        best_i, best_d2 = -1, self.r * self.r
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for i in d.get((gx, gy), ()):
                    px, py = pts[i]
                    dd = (px - x) ** 2 + (py - y) ** 2
                    if dd <= best_d2:
                        best_d2, best_i = dd, i
        if best_i >= 0:
            return best_i
        i = len(pts); pts.append([float(x), float(y)])
        d.setdefault((cx, cy), []).append(i)
        return i


@torch.inference_mode()
def match_loftr(paths, pairs, max_dim, conf_thr, min_matches):
    loftr = KF.LoFTR(pretrained="outdoor").to(DEV).eval()
    imgs, scales, hws = {}, {}, {}
    for p in paths:
        imgs[p.name], scales[p.name], hws[p.name] = load_gray(p, max_dim)
    agg = KeypointAggregator()
    pair_matches = {}
    for a, b in pairs:
        out = loftr({"image0": imgs[a], "image1": imgs[b]})
        conf = out["confidence"]
        keep = conf >= conf_thr
        k0 = out["keypoints0"][keep].cpu().numpy()
        k1 = out["keypoints1"][keep].cpu().numpy()
        if len(k0) < min_matches:
            continue
        sa, sb = scales[a], scales[b]
        m = []
        for (x0, y0), (x1, y1) in zip(k0, k1):
            i0 = agg.add(a, x0 * sa[0], y0 * sa[1])
            i1 = agg.add(b, x1 * sb[0], y1 * sb[1])
            m.append((i0, i1))
        pair_matches[(a, b)] = np.array(m, dtype=np.uint32)
    del loftr
    torch.cuda.empty_cache()
    return agg, pair_matches, hws


def build_db(agg, pair_matches, hws, db_path):
    if db_path.exists():
        db_path.unlink()
    db = pycolmap.Database.open(str(db_path))
    ids = {}
    for name, pts in agg.pts.items():
        H, W = hws[name]
        cam = pycolmap.Camera(model="SIMPLE_RADIAL", width=W, height=H,
                              params=[1.2 * max(H, W), W / 2, H / 2, 0.0])
        cid = db.write_camera(cam, use_camera_id=False)
        iid = db.write_image(pycolmap.Image(name=name, camera_id=cid), use_image_id=False)
        ids[name] = iid
        db.write_keypoints(iid, np.array(pts, dtype=np.float32))
    pairs_txt = db_path.parent / "pairs.txt"
    with open(pairs_txt, "w") as fh:
        for (a, b), m in pair_matches.items():
            if a in ids and b in ids:
                db.write_matches(ids[a], ids[b], m)
                fh.write(f"{a} {b}\n")
    db.close()
    return pairs_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir"); ap.add_argument("work_dir")
    ap.add_argument("--max_dim", type=int, default=840)
    ap.add_argument("--conf_thr", type=float, default=0.4)
    ap.add_argument("--min_matches", type=int, default=15)
    ap.add_argument("--ba", action="store_true")
    ap.add_argument("--score_train", default=None)
    ap.add_argument("--tag", default="loftr")
    args = ap.parse_args()

    scene = Path(args.scene_dir); work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in scene.iterdir() if p.suffix.lower() in IMG_EXT)
    t0 = time.time()
    pairs = list(combinations([p.name for p in paths], 2))
    agg, pm, hws = match_loftr(paths, pairs, args.max_dim, args.conf_thr, args.min_matches)

    db_path = work / "database.db"
    pairs_txt = build_db(agg, pm, hws, db_path)
    pycolmap.verify_matches(str(db_path), str(pairs_txt))
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = True
    maps = pycolmap.incremental_mapping(str(db_path), str(scene), str(work / "sparse"), opts)
    if args.ba:
        for rec in maps.values():
            pycolmap.bundle_adjustment(rec, pycolmap.BundleAdjustmentOptions())

    preds, total = {}, 0
    for rec in maps.values():
        total += rec.num_reg_images()
        for _, img in rec.images.items():
            if img.has_pose:
                pose = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
                preds.setdefault(img.name, (pose.rotation.matrix(), np.asarray(pose.translation)))
    line = f"[RESULT] {args.tag}: reg={total/max(len(paths),1):.0%} ({total}/{len(paths)}) pairs={len(pm)} time={time.time()-t0:.0f}s"
    if args.score_train:
        import imc_metric
        root = scene.parents[1]
        res = imc_metric.score_dataset(preds, root / "train_labels.csv",
                                       root / "train_thresholds.csv", args.score_train)
        line += (f" rot_mAA={np.mean([m['rot_mAA'] for m in res.values()]):.3f}"
                 f" trans_mAA={np.mean([m['trans_mAA'] for m in res.values()]):.3f}")
    print(line)


if __name__ == "__main__":
    main()
