"""RoMa (dense, SOTA) SfM — Tier 3 matcher for the hardest scenes.

RoMa v2 is currently the strongest dense feature matcher for relative pose. Like
LoFTR it is detector-free, so we reuse the per-image keypoint-aggregation trick
(quantize matched points to an integer grid → one keypoint set per image).

    python run_roma.py <scene_dir> <work_dir> [--num 2000] [--score_train NAME]

RoMa is heavy (~0.3-0.6s/pair); intended for small hard scenes, not the big ones.
"""
import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pycolmap
import torch
from PIL import Image

from run_loftr import KeypointAggregator, build_db  # reuse aggregation + DB import

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXT = {".png", ".jpg", ".jpeg"}


@torch.inference_mode()
def match_roma(paths, pairs, num, conf_thr, min_matches, indoor=False, merge_radius=0.0):
    from romatch import roma_indoor, roma_outdoor
    builder = roma_indoor if indoor else roma_outdoor
    model = builder(device=DEV, use_custom_corr=False)  # pure-torch corr (no CUDA build)
    hws = {}
    for p in paths:
        with Image.open(p) as im:
            hws[p.name] = (im.height, im.width)
    by_name = {p.name: p for p in paths}
    agg = KeypointAggregator(merge_radius)
    pair_matches = {}
    for a, b in pairs:
        Ha, Wa = hws[a]; Hb, Wb = hws[b]
        warp, cert = model.match(str(by_name[a]), str(by_name[b]), device=DEV)
        matches, certainty = model.sample(warp, cert, num=num)
        kA, kB = model.to_pixel_coordinates(matches, Ha, Wa, Hb, Wb)
        keep = certainty >= conf_thr
        kA = kA[keep].cpu().numpy(); kB = kB[keep].cpu().numpy()
        if len(kA) < min_matches:
            continue
        m = [(agg.add(a, x0, y0), agg.add(b, x1, y1))
             for (x0, y0), (x1, y1) in zip(kA, kB)]
        pair_matches[(a, b)] = np.array(m, dtype=np.uint32)
    del model
    torch.cuda.empty_cache()
    return agg, pair_matches, hws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir"); ap.add_argument("work_dir")
    ap.add_argument("--num", type=int, default=2000)
    ap.add_argument("--conf_thr", type=float, default=0.5)
    ap.add_argument("--min_matches", type=int, default=15)
    ap.add_argument("--ba", action="store_true")
    ap.add_argument("--indoor", action="store_true", help="use roma_indoor (for indoor scenes like stairs)")
    ap.add_argument("--merge_radius", type=float, default=0.0, help="radius-merge dense keypoints for track consistency (try 3)")
    ap.add_argument("--score_train", default=None)
    ap.add_argument("--tag", default="roma")
    args = ap.parse_args()

    scene = Path(args.scene_dir); work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in scene.iterdir() if p.suffix.lower() in IMG_EXT)
    t0 = time.time()
    pairs = list(combinations([p.name for p in paths], 2))
    agg, pm, hws = match_roma(paths, pairs, args.num, args.conf_thr, args.min_matches,
                              args.indoor, args.merge_radius)

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
                 f" trans_mAA={np.mean([m['trans_mAA'] for m in res.values()]):.3f}"
                 f" COMBINED={np.mean([m.get('combined_mAA', 0.0) for m in res.values()]):.3f}")
    print(line)


if __name__ == "__main__":
    main()
