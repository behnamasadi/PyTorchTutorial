#!/usr/bin/env python
"""Evaluate VGGT-Omega on IMC train datasets with the local mAA scorer.

Feed-forward: one forward pass over all images of a dataset, no matching, no
SfM, no bundle adjustment. Emits the same ``[RESULT]`` line as the MASt3R and
DISK+LightGlue experiments so numbers are directly comparable.

    python vggt_omega_eval.py --dataset stairs
    python vggt_omega_eval.py --dataset imc2023_haiper --resolution 512
    python vggt_omega_eval.py --dataset ETs --export-colmap out/ETs_colmap

Two caveats when reading the score:

1. **Every image is registered by construction.** The model returns a pose for
   each input frame, so reg_rate is always 100%. Unlike COLMAP it cannot decline
   to place an image, so a bad pose lands in the score instead of being dropped.
2. **Clustering is oracle.** ``score_dataset`` groups by ground-truth scene, so
   a multi-scene dataset is scored as if the clustering were perfect. Real IMC
   requires predicting the clusters too. Single-scene datasets (amy_gardens) and
   per-scene runs (--per-scene) avoid this.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "extracted"
IMG_EXT = {".png", ".jpg", ".jpeg", ".JPG", ".PNG"}


def load_model(checkpoint, device: str):
    """checkpoint=None -> random init, for validating plumbing without weights.

    Weights stay fp32 on purpose: the camera head runs under
    autocast(enabled=False) and rejects bf16 parameters. The forward pass already
    autocasts the backbone, and activations — not the 4.5 GB of weights — are what
    dominate memory, so resolution is the lever for fitting more frames.
    """
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega().to(device).eval()
    if checkpoint is None:
        print("!! RANDOM WEIGHTS — plumbing check only, scores are meaningless")
    else:
        state = torch.load(str(checkpoint), map_location="cpu")
        state = state.get("model", state)
        model.load_state_dict(state)
    return model


def fetch_checkpoint(name: str = "vggt_omega_1b_512.pt") -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download("facebook/VGGT-Omega", name))


def run_images(model, image_paths, resolution: int, device: str):
    """One forward pass -> {image_name: (R 3x3, t 3)} cam_from_world, OpenCV."""
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    images = load_and_preprocess_images(
        [str(p) for p in image_paths], image_resolution=resolution
    ).to(device)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        predictions = model(images)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"], predictions["images"].shape[-2:]
    )
    extrinsics = extrinsics[0].float().cpu().numpy()  # (N, 3, 4)
    intrinsics = intrinsics[0].float().cpu().numpy()  # (N, 3, 3)

    # A non-finite pose is treated as *unregistered* rather than propagated into
    # the Umeyama fit, where a single NaN makes the SVD fail for the whole scene.
    preds, dropped = {}, 0
    for i, p in enumerate(image_paths):
        if not np.isfinite(extrinsics[i]).all():
            dropped += 1
            continue
        preds[p.name] = (extrinsics[i, :, :3], extrinsics[i, :, 3])
    if dropped:
        print(f"  !! {dropped}/{len(image_paths)} non-finite poses dropped (unregistered)")
    return preds, intrinsics, elapsed, peak_gb, predictions


def export_colmap(out_dir: Path, preds, intrinsics, image_paths, image_hw):
    """Write cameras.txt / images.txt / points3D.txt (poses only, empty points).

    vggt-omega ships no COLMAP export, so this is the hand-off point for
    triangulation + bundle adjustment in pycolmap.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    from vggt_omega.utils.rotation import mat_to_quat

    H, W = image_hw
    with open(out_dir / "cameras.txt", "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i in range(len(image_paths)):
            K = intrinsics[i]
            f.write(
                f"{i + 1} PINHOLE {W} {H} {K[0, 0]:.6f} {K[1, 1]:.6f} "
                f"{K[0, 2]:.6f} {K[1, 2]:.6f}\n"
            )

    with open(out_dir / "images.txt", "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for i, p in enumerate(image_paths):
            R, t = preds[p.name]
            q = mat_to_quat(torch.from_numpy(R)[None].float())[0].numpy()  # (x,y,z,w)
            qw, qx, qy, qz = q[3], q[0], q[1], q[2]
            f.write(
                f"{i + 1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {i + 1} {p.name}\n\n"
            )

    (out_dir / "points3D.txt").write_text("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
    print(f"  COLMAP model -> {out_dir}  (poses only; run point_triangulator + BA next)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. stairs, ETs, imc2023_haiper")
    ap.add_argument("--split", default="train")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--checkpoint", default=None, help="local .pt; default downloads from HF")
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames (memory)")
    ap.add_argument("--per-scene", action="store_true",
                    help="run each GT scene separately (no oracle-clustering advantage,"
                         " and far less memory)")
    ap.add_argument("--export-colmap", default=None, help="output dir for a COLMAP model")
    ap.add_argument("--save-npz", default=None)
    ap.add_argument("--random-weights", action="store_true",
                    help="skip the gated download; validate the pipeline end-to-end "
                         "with an untrained model (scores are meaningless)")
    args = ap.parse_args()

    import pandas as pd
    from imc_metric import score_dataset

    device = "cuda"
    scene_dir = DATA / args.split / args.dataset
    if not scene_dir.is_dir():
        raise SystemExit(f"not found: {scene_dir}")

    labels_csv = DATA / "train_labels.csv"
    thresholds_csv = DATA / "train_thresholds.csv"

    all_imgs = sorted(p for p in scene_dir.iterdir() if p.suffix in IMG_EXT)
    if not all_imgs:
        raise SystemExit(f"no images in {scene_dir}")

    if args.random_weights:
        ckpt = None
    else:
        ckpt = Path(args.checkpoint) if args.checkpoint else fetch_checkpoint()
    print(f"checkpoint: {ckpt}")
    model = load_model(ckpt, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"model: {n_params:.2f}B params | dataset={args.dataset} "
          f"images={len(all_imgs)} resolution={args.resolution}")

    groups = {}
    if args.per_scene:
        gt = pd.read_csv(labels_csv)
        gt = gt[gt.dataset == args.dataset]
        for scene, g in gt.groupby("scene"):
            if scene == "outliers":
                continue
            names = set(g.image)
            sel = [p for p in all_imgs if p.name in names]
            if sel:
                groups[scene] = sel
    else:
        groups["__all__"] = all_imgs

    preds, total_time, peak = {}, 0.0, 0.0
    last = None
    for gname, paths in groups.items():
        if args.max_frames:
            paths = paths[: args.max_frames]
        p, K, dt, gb, raw = run_images(model, paths, args.resolution, device)
        preds.update(p)
        total_time += dt
        peak = max(peak, gb)
        last = (p, K, paths, raw["images"].shape[-2:])
        tag = "" if gname == "__all__" else f" [{gname}]"
        print(f"  forward{tag}: {len(paths)} frames in {dt:.1f}s "
              f"({len(paths) / dt:.1f} fps) peak {gb:.2f} GB")

    if not preds:
        raise SystemExit("no finite poses predicted — nothing to score")
    try:
        res = score_dataset(preds, labels_csv, thresholds_csv, args.dataset)
    except np.linalg.LinAlgError as e:
        raise SystemExit(
            f"Sim(3) alignment failed ({e}). Degenerate camera-centre configuration "
            f"— expected with --random-weights, investigate if it happens with real ones."
        )

    print(f"\n{'scene':<28} {'reg':>9} {'rot_mAA':>8} {'trans_mAA':>10} {'combined':>9}")
    for scene, r in sorted(res.items()):
        print(f"{scene:<28} {r['registered']:>4}/{r['n']:<4} {r['rot_mAA']:>8.3f} "
              f"{r['trans_mAA']:>10.3f} {r['combined_mAA']:>9.3f}")

    mean = lambda k: float(np.mean([r[k] for r in res.values()])) if res else 0.0
    print(f"\n[RESULT] vggt-omega: n={len(all_imgs)} pairs=0 time={total_time:.0f}s "
          f"rot_mAA={mean('rot_mAA'):.3f} trans_mAA={mean('trans_mAA'):.3f} "
          f"COMBINED={mean('combined_mAA'):.3f} peak_mem={peak:.1f}GB "
          f"res={args.resolution}")

    if args.save_npz:
        names = sorted(preds)
        np.savez(
            args.save_npz,
            names=np.array(names),
            extrinsics=np.stack([np.concatenate([preds[n][0], preds[n][1][:, None]], 1)
                                 for n in names]),
        )
    if args.export_colmap and last:
        p, K, paths, hw = last
        export_colmap(Path(args.export_colmap), p, K, paths, hw)


if __name__ == "__main__":
    main()
