"""MASt3R-SfM — 3D-grounded global alignment for the ACCURACY-limited dead scenes.

Every coverage lever (dense matching, ensemble, wider retrieval) raised registration
but left combined_mAA flat on stairs/vineyard/amy — the bottleneck is pose ACCURACY,
not coverage. MASt3R regresses metric 3D geometry and does a global 3D optimization
(sparse_global_alignment), which is the one thing that can improve translation on
repetitive/low-texture scenes where triangulation is ambiguous.

    python run_mast3r.py <scene_dir> <work_dir> [--scene_graph complete|swin-5] [--score_train NAME]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/work/mast3r")
from mast3r.model import AsymmetricMASt3R          # noqa: E402
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment  # noqa: E402
from mast3r.image_pairs import make_pairs          # noqa: E402
from dust3r.utils.image import load_images         # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg"}
CKPT = "/work/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("work_dir")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--scene_graph", default="complete", help="complete (O(n^2)) | swin-N (windowed)")
    ap.add_argument("--score_train", default=None)
    ap.add_argument("--tag", default="mast3r")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    scene_dir = Path(args.scene_dir)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    paths = sorted(str(p) for p in scene_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    names = [Path(p).name for p in paths]

    t0 = time.time()
    model = AsymmetricMASt3R.from_pretrained(CKPT).to(dev).eval()
    images = load_images(paths, size=args.size, verbose=False)
    pairs = make_pairs(images, scene_graph=args.scene_graph, symmetrize=True)
    scene = sparse_global_alignment(paths, pairs, str(work / "cache"), model, device=dev)
    cam2w = scene.get_im_poses().detach().cpu().numpy()  # (N,4,4) camera-to-world

    preds = {}
    for i, name in enumerate(names):
        R_c2w = cam2w[i, :3, :3]
        t_c2w = cam2w[i, :3, 3]
        preds[name] = (R_c2w.T, -R_c2w.T @ t_c2w)  # cam_from_world (R, t)

    line = f"[RESULT] {args.tag}: n={len(names)} pairs={len(pairs)} time={time.time()-t0:.0f}s"
    if args.score_train:
        import imc_metric
        root = scene_dir.parents[1]
        res = imc_metric.score_dataset(preds, root / "train_labels.csv",
                                       root / "train_thresholds.csv", args.score_train)
        if res:
            line += (f" rot_mAA={np.mean([m['rot_mAA'] for m in res.values()]):.3f}"
                     f" trans_mAA={np.mean([m['trans_mAA'] for m in res.values()]):.3f}"
                     f" COMBINED={np.mean([m.get('combined_mAA', 0.0) for m in res.values()]):.3f}")
    print(line, flush=True)


if __name__ == "__main__":
    main()
