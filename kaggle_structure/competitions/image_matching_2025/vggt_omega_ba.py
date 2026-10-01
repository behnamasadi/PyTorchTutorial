#!/usr/bin/env python
"""BA polish on top of VGGT-Omega poses: the feed-forward-init + classical-refine hybrid.

Per scene: Omega forward pass -> poses + intrinsics -> SIFT extract + exhaustive
match (pycolmap) -> triangulate with Omega poses fixed as init -> global bundle
adjustment -> rescore. Reports the scene's mAA before and after BA so the delta
is the experiment.

    python vggt_omega_ba.py --dataset ETs --scene ET
    python vggt_omega_ba.py --dataset imc2023_haiper            # all scenes

Intrinsics note: Omega predicts K at its resized resolution (~624x432); SIFT
runs on original pixels. K is rescaled by the per-image size ratio and the
principal point reset to the original center — approximate on purpose, BA
refines focal. Principal point stays fixed (refining it with weak geometry
diverges).
"""

import argparse
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "extracted"
WORK = HERE / "work_omega_ba"  # NOT work/ — that dir is root-owned (docker legacy)
IMG_EXT = {".png", ".jpg", ".jpeg", ".JPG", ".PNG"}


def scene_images(dataset, scene):
    import pandas as pd

    gt = pd.read_csv(DATA / "train_labels.csv")
    g = gt[(gt.dataset == dataset) & (gt.scene == scene)]
    src = DATA / "train" / dataset
    return [src / n for n in sorted(g.image) if (src / n).exists()]


def write_ref_model(out_dir, preds, K_pred, pred_hw, image_paths):
    """COLMAP text model: Omega poses + per-image PINHOLE K scaled to original pixels."""
    from PIL import Image as PILImage
    from vggt_omega.utils.rotation import mat_to_quat
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    ph, pw = pred_hw
    cams, imgs = [], []
    for i, p in enumerate(image_paths):
        W, H = PILImage.open(p).size
        s = max(W, H) / max(pw, ph)  # aspect-preserving resize factor
        fx, fy = K_pred[i][0, 0] * s, K_pred[i][1, 1] * s
        cams.append((i + 1, W, H, fx, fy))
        R, t = preds[p.name]
        q = mat_to_quat(torch.from_numpy(np.ascontiguousarray(R))[None].float())[0].numpy()
        imgs.append((i + 1, q, t, p.name))

    with open(out_dir / "cameras.txt", "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cid, W, H, fx, fy in cams:
            f.write(f"{cid} PINHOLE {W} {H} {fx:.4f} {fy:.4f} {W/2:.2f} {H/2:.2f}\n")
    with open(out_dir / "images.txt", "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for iid, q, t, name in imgs:
            qx, qy, qz, qw = q
            f.write(f"{iid} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {iid} {name}\n\n")
    (out_dir / "points3D.txt").write_text("")


def align_db_ids(db_path, ref_dir):
    """Rewrite the text model so image/camera ids match what extraction put in the db."""
    con = sqlite3.connect(db_path)
    db_ids = dict(con.execute("SELECT name, image_id FROM images").fetchall())
    db_cam = dict(con.execute("SELECT image_id, camera_id FROM images"
                              " UNION SELECT i.image_id, i.camera_id FROM images i").fetchall())
    con.close()

    cams = {}
    for line in (ref_dir / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        cams[int(parts[0])] = parts[1:]

    out_cams, out_imgs = [], []
    lines = [l for l in (ref_dir / "images.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    for line in lines:
        parts = line.split()
        old_iid, name = int(parts[0]), parts[-1]
        if name not in db_ids:
            continue
        new_iid = db_ids[name]
        new_cid = db_cam[new_iid]
        out_imgs.append(f"{new_iid} {' '.join(parts[1:8])} {new_cid} {name}\n\n")
        out_cams.append(f"{new_cid} {' '.join(cams[old_iid])}\n")

    with open(ref_dir / "cameras.txt", "w") as f:
        f.writelines(sorted(set(out_cams), key=lambda s: int(s.split()[0])))
    with open(ref_dir / "images.txt", "w") as f:
        f.writelines(out_imgs)


def score_scene(preds, dataset, scene):
    from imc_metric import score_dataset

    res = score_dataset(preds, DATA / "train_labels.csv",
                        DATA / "train_thresholds.csv", dataset)
    return res[scene]


def run_scene(model, dataset, scene, resolution, args):
    import pycolmap
    from vggt_omega_eval import run_images

    paths = scene_images(dataset, scene)
    if len(paths) < 3:
        print(f"[{scene}] <3 images, skip")
        return None
    print(f"\n=== {dataset}/{scene}: {len(paths)} images ===")

    preds, K_pred, dt, _gb, raw = run_images(model, paths, resolution, "cuda")
    pred_hw = raw["images"].shape[-2:]
    before = score_scene(preds, dataset, scene)
    print(f"  omega only : rot {before['rot_mAA']:.3f} trans {before['trans_mAA']:.3f} "
          f"combined {before['combined_mAA']:.3f}  ({dt:.1f}s fwd)")

    w = WORK / dataset / scene
    if w.exists():
        shutil.rmtree(w)
    img_dir = w / "images"
    img_dir.mkdir(parents=True)
    for p in paths:
        (img_dir / p.name).symlink_to(p.resolve())

    ref = w / "ref"
    write_ref_model(ref, preds, K_pred, pred_hw, paths)

    db = w / "db.db"
    t0 = time.time()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    pycolmap.extract_features(str(db), str(img_dir),
                              camera_mode=pycolmap.CameraMode.PER_IMAGE,
                              reader_options=reader)
    pycolmap.match_exhaustive(str(db))
    align_db_ids(db, ref)

    recon = pycolmap.Reconstruction(str(ref))
    tri_out = w / "triangulated"
    tri_out.mkdir(exist_ok=True)
    recon = pycolmap.triangulate_points(recon, str(db), str(img_dir), str(tri_out))
    n_pts = len(recon.points3D)

    ba_opts = pycolmap.BundleAdjustmentOptions()
    ba_opts.refine_principal_point = False
    pycolmap.bundle_adjustment(recon, ba_opts)
    t_cls = time.time() - t0

    preds_ba = {}
    for img in recon.images.values():
        cfw = img.cam_from_world()
        preds_ba[img.name] = (cfw.rotation.matrix(), np.array(cfw.translation))
    after = score_scene(preds_ba, dataset, scene)
    print(f"  omega + BA : rot {after['rot_mAA']:.3f} trans {after['trans_mAA']:.3f} "
          f"combined {after['combined_mAA']:.3f}  "
          f"({n_pts} pts, {len(preds_ba)}/{len(paths)} reg, {t_cls:.0f}s classical)")
    return {"scene": scene, "before": before, "after": after}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scene", default=None, help="default: all non-outlier scenes")
    ap.add_argument("--resolution", type=int, default=512)
    args = ap.parse_args()

    import pandas as pd
    from vggt_omega_eval import fetch_checkpoint, load_model

    model = load_model(fetch_checkpoint(), "cuda")

    if args.scene:
        scenes = [args.scene]
    else:
        gt = pd.read_csv(DATA / "train_labels.csv")
        scenes = [s for s in gt[gt.dataset == args.dataset].scene.unique()
                  if s != "outliers"]

    results = [r for s in scenes if (r := run_scene(model, args.dataset, s,
                                                    args.resolution, args))]
    if results:
        db = float(np.mean([r["before"]["combined_mAA"] for r in results]))
        da = float(np.mean([r["after"]["combined_mAA"] for r in results]))
        print(f"\n[RESULT] omega+BA {args.dataset}: combined {db:.3f} -> {da:.3f} "
              f"(delta {da - db:+.3f}) over {len(results)} scenes")


if __name__ == "__main__":
    main()
