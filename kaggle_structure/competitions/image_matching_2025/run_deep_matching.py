"""Deep-matching SfM baseline (Tier 1): DISK + LightGlue -> pycolmap.

GPU deep features/matches imported into a COLMAP database, geometrically verified,
then incremental mapping for precise poses. This replaces the dead SIFT stage.

    python run_deep_matching.py <scene_dir> <work_dir> [--max_dim 1024] [--max_kp 2048]

Reports registered images + per-image (R, t). Designed so LoFTR / ALIKED / RoMa
can be added as extra matchers whose matches concatenate into the same DB (ensemble).
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


def load_gray_rgb(path: Path, max_dim: int):
    """Return (rgb 1x3xHxW, gray 1x1xHxW) on device, scaled so max side <= max_dim, + scale factor."""
    img = K.io.load_image(str(path), K.io.ImageLoadType.RGB32, device=DEV)[None]  # 1x3xHxW in [0,1]
    h, w = img.shape[-2:]
    s = min(1.0, max_dim / max(h, w))
    if s < 1.0:
        img = K.geometry.resize(img, (round(h * s), round(w * s)), antialias=True)
    return img, K.color.rgb_to_grayscale(img), s, (h, w)


@torch.inference_mode()
def extract_disk(paths, max_dim, max_kp):
    disk = KF.DISK.from_pretrained("depth").to(DEV).eval()
    feats = {}
    for p in paths:
        rgb, _, s, (H, W) = load_gray_rgb(p, max_dim)
        f = disk(rgb, max_kp, pad_if_not_divisible=True)[0]
        kps = f.keypoints / s  # back to original-resolution pixel coords
        feats[p.name] = {"kps": kps, "desc": f.descriptors, "hw": (H, W)}
    del disk
    torch.cuda.empty_cache()
    return feats


@torch.inference_mode()
def match_all(feats, pairs, min_matches):
    lg = KF.LightGlueMatcher("disk").to(DEV).eval()
    out = {}
    for a, b in pairs:
        fa, fb = feats[a], feats[b]
        lafs_a = KF.laf_from_center_scale_ori(fa["kps"][None].float())
        lafs_b = KF.laf_from_center_scale_ori(fb["kps"][None].float())
        _, idxs = lg(fa["desc"], fb["desc"], lafs_a, lafs_b,
                     hw1=torch.tensor(fa["hw"]), hw2=torch.tensor(fb["hw"]))
        if len(idxs) >= min_matches:
            out[(a, b)] = idxs.cpu().numpy().astype(np.uint32)
    return out


def build_db(feats, matches, image_dir: Path, db_path: Path):
    if db_path.exists():
        db_path.unlink()
    db = pycolmap.Database.open(str(db_path))
    name_to_id = {}
    for name, f in feats.items():
        H, W = f["hw"]
        cam = pycolmap.Camera(model="SIMPLE_RADIAL", width=W, height=H,
                              params=[1.2 * max(H, W), W / 2, H / 2, 0.0])
        cam_id = db.write_camera(cam, use_camera_id=False)
        img = pycolmap.Image(name=name, camera_id=cam_id)
        img_id = db.write_image(img, use_image_id=False)
        name_to_id[name] = img_id
        db.write_keypoints(img_id, f["kps"].cpu().numpy().astype(np.float32))
    pairs_txt = db_path.parent / "pairs.txt"
    with open(pairs_txt, "w") as fh:
        for (a, b), m in matches.items():
            db.write_matches(name_to_id[a], name_to_id[b], m)
            fh.write(f"{a} {b}\n")
    db.close()
    return pairs_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir"); ap.add_argument("work_dir")
    ap.add_argument("--max_dim", type=int, default=1024)
    ap.add_argument("--max_kp", type=int, default=2048)
    ap.add_argument("--min_matches", type=int, default=15)
    ap.add_argument("--score_train", default=None,
                    help="dataset name to score against train_labels.csv")
    args = ap.parse_args()

    scene = Path(args.scene_dir); work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in scene.iterdir() if p.suffix.lower() in IMG_EXT)
    print(f"[deep] scene={scene.name} images={len(paths)} device={DEV}")

    t0 = time.time()
    feats = extract_disk(paths, args.max_dim, args.max_kp)
    print(f"[deep] DISK features done ({time.time()-t0:.1f}s)")

    pairs = list(combinations([p.name for p in paths], 2))  # exhaustive (small scenes)
    t0 = time.time()
    matches = match_all(feats, pairs, args.min_matches)
    print(f"[deep] LightGlue matched {len(matches)}/{len(pairs)} pairs ({time.time()-t0:.1f}s)")

    db_path = work / "database.db"
    pairs_txt = build_db(feats, matches, scene, db_path)
    t0 = time.time()
    pycolmap.verify_matches(str(db_path), str(pairs_txt))
    print(f"[deep] geometric verification done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    maps = pycolmap.incremental_mapping(str(db_path), str(scene), str(work / "sparse"))
    print(f"[deep] mapping done ({time.time()-t0:.1f}s) models={len(maps)}")

    preds, total = {}, 0
    for idx, rec in maps.items():
        n = rec.num_reg_images(); total += n
        print(f"  model {idx}: registered {n}/{len(paths)}")
        for _, img in rec.images.items():
            if not img.has_pose:
                continue
            pose = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
            preds.setdefault(img.name, (pose.rotation.matrix(), np.asarray(pose.translation)))
    print(f"[deep] TOTAL registered {total}/{len(paths)} ({total/max(len(paths),1):.0%})")

    if args.score_train:
        import imc_metric
        data_root = scene.parents[1]  # data/extracted
        res = imc_metric.score_dataset(preds, data_root / "train_labels.csv",
                                       data_root / "train_thresholds.csv", args.score_train)
        print(f"[score] dataset={args.score_train}")
        for sc, m in res.items():
            print(f"  {sc:24s} reg {m['registered']:>3}/{m['n']:<3} ({m['reg_rate']:.0%})  "
                  f"rot_mAA={m['rot_mAA']:.3f}  trans_mAA={m['trans_mAA']:.3f}  "
                  f"med_rot={m.get('median_rot_err','-')}°")
        import numpy as _np
        print(f"[score] MEAN trans_mAA={_np.mean([m['trans_mAA'] for m in res.values()]):.3f}  "
              f"rot_mAA={_np.mean([m['rot_mAA'] for m in res.values()]):.3f}")


if __name__ == "__main__":
    main()
