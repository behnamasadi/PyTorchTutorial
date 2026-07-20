"""IMC 2025-ongoing — offline inference notebook (code competition).

Runs on Kaggle GPU with NO internet. Dependencies come from the attached dataset
`imc-2026-deps-min` (pycolmap wheel + DISK/LightGlue weights) and
`imc-torch251-cu121` (P100-compatible torch). kornia is preinstalled on Kaggle.

Pipeline (per test dataset): DISK features -> retrieval-restricted top-k pairing
(global descriptors pooled from the DISK features -> NO extra model / internet) ->
LightGlue match -> tuned geometric verification -> pycolmap incremental mapping
(models = predicted scenes) -> submission.csv.

P0 fixes vs the earlier version (which hung PENDING on the hidden test set):
  1. Retrieval top-k pairing replaces exhaustive O(n^2) matching (the timeout cause).
  2. Global wall-clock budget: submission.csv is (re)written after EVERY dataset and
     datasets run small-first, so a kill at the time limit still leaves a valid,
     partially-filled submission (a timeout scores nothing at all).
  3. GPU is validated ONCE up front; no more per-scene "rerun the whole dataset on
     CPU" loop that guaranteed a timeout on a silent CUDA/arch mismatch.
  4. Tuned geometric verification (the one confirmed local win over the DISK baseline).

This .py is the single notebook cell; push via `kaggle kernels push`.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

START = time.time()
# Conservative wall-clock budget. Kaggle Community code comps allow up to ~9-12h;
# we stop launching new heavy work past this so submission.csv is always flushed
# well before the hard kill. Tune down if the exact limit is known to be lower.
TIME_BUDGET_SEC = float(os.environ.get("IMC_TIME_BUDGET", 8.5 * 3600))


def elapsed():
    return time.time() - START


def remaining():
    return TIME_BUDGET_SEC - elapsed()


DEPS = Path("/kaggle/input/imc-2026-deps-min")

# Show what's actually mounted (diagnostics), then find the competition dir robustly.
print("=== /kaggle/input tree ===", flush=True)
for p in sorted(Path("/kaggle/input").rglob("*"))[:60]:
    print(" ", p, flush=True)
_ss = list(Path("/kaggle/input").rglob("sample_submission.csv"))
print("sample_submission.csv found at:", _ss, flush=True)
COMP = _ss[0].parent if _ss else Path("/kaggle/input")


TORCH_DIR = Path("/kaggle/input/imc-torch251-cu121")


def setup_offline():
    import glob
    import shutil
    import zipfile
    # 0. P100 fix: Kaggle's torch dropped Pascal sm_60 -> install torch 2.5.1+cu121 offline.
    #    Kaggle STRIPS the '+' from wheel filenames (torch-2.5.1cu121-...), which pip rejects as an
    #    invalid version. Fix: copy wheels into a tmp dir restoring '+cu' (2.5.1cu121 -> 2.5.1+cu121),
    #    then install from there with --find-links resolving deps.
    import re
    if TORCH_DIR.exists():
        twd = Path("/kaggle/tmp/torch_wheels"); twd.mkdir(parents=True, exist_ok=True)
        for w in TORCH_DIR.glob("*.whl"):
            fixed = re.sub(r"(\d)(cu\d)", r"\1+\2", w.name)  # restore '+' before 'cuNN'
            shutil.copy(w, twd / fixed)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                            "--find-links", str(twd), "torch==2.5.1", "torchvision==0.20.1"])
        print("torch 2.5.1 offline install rc:", r.returncode, flush=True)
    tmp = Path("/kaggle/tmp/deps")
    tmp.mkdir(parents=True, exist_ok=True)
    # deps dir may contain zipped subfolders (weights.zip / wheels.zip) — extract them
    for z in DEPS.rglob("*.zip"):
        try:
            zipfile.ZipFile(z).extractall(tmp)
        except Exception as e:
            print("unzip skip", z, e, flush=True)
    roots = [DEPS, tmp]
    # find + install the pycolmap wheel wherever it landed
    whl = next((w for r in roots for w in r.rglob("pycolmap*.whl")), None)
    print("pycolmap wheel:", whl, flush=True)
    if whl:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                        "--no-deps", str(whl)], check=True)
    # copy all model weights into the torch hub cache (kornia loads offline from here)
    cache = Path("/root/.cache/torch/hub/checkpoints")
    cache.mkdir(parents=True, exist_ok=True)
    for r in roots:
        for w in r.rglob("*"):
            if w.is_file() and ("pth" in w.name.lower() or w.suffix in (".ckpt", ".safetensors")):
                dst = cache / w.name
                if not dst.exists():
                    shutil.copy(w, dst)
    print("cached weights:", sorted(p.name for p in cache.glob("*")), flush=True)


setup_offline()

import torch                     # noqa: E402
print("torch", torch.__version__, "| cuda avail", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0),
          "| arch_list", torch.cuda.get_arch_list(), flush=True)
import kornia as K               # noqa: E402
import kornia.feature as KF      # noqa: E402
import pycolmap                  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg"}
# 8192 kp (up from 4096) — measured locally to 3.2× the combined mAA on the weak
# heritage scene (0.046→0.149) with no regression on easy scenes. We were feature-starved.
MAX_KP, MAX_DIM, MIN_MATCHES = 8192, 1280, 15
TOPK = 30            # retrieval neighbours per image (pairs pruned to ~n*k, not n^2)
EXHAUSTIVE_MAX = 50  # small scenes: match every pair (retrieval not worth it)


def gpu_ok():
    """Validate the GPU once: run a tiny DISK forward. A silent CUDA/arch mismatch
    (P100 fix failed) surfaces HERE, not 8 hours into per-scene CPU fallbacks."""
    if not torch.cuda.is_available():
        return False
    try:
        disk = KF.DISK.from_pretrained("depth").to("cuda").eval()
        with torch.inference_mode():
            disk(torch.zeros(1, 3, 64, 64, device="cuda"), 128, pad_if_not_divisible=True)
        del disk
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print("GPU validation FAILED -> CPU:", str(e)[:120], flush=True)
        return False


DEV = torch.device("cuda" if gpu_ok() else "cpu")
print("compute device:", DEV, flush=True)


@torch.inference_mode()
def extract(paths):
    disk = KF.DISK.from_pretrained("depth").to(DEV).eval()
    feats = {}
    for p in paths:
        img = K.io.load_image(str(p), K.io.ImageLoadType.RGB32, device=DEV)[None]
        h, w = img.shape[-2:]
        s = min(1.0, MAX_DIM / max(h, w))
        if s < 1.0:
            img = K.geometry.resize(img, (round(h * s), round(w * s)), antialias=True)
        f = disk(img, MAX_KP, pad_if_not_divisible=True)[0]
        # global descriptor for retrieval: mean+max pool of the (already L2-normed) DISK
        # descriptors -> a cheap, model-free covisibility proxy from the same features
        # that will be matched. Prunes pairs without any extra weights / internet.
        d = f.descriptors
        if d.numel():
            g = torch.cat([d.mean(0), d.amax(0)]).float().cpu().numpy()
        else:
            g = np.zeros(2 * (d.shape[1] if d.ndim == 2 else 128), dtype=np.float32)
        feats[p.name] = {"kps": f.keypoints / s, "desc": d, "hw": (h, w), "gdesc": g}
    del disk
    torch.cuda.empty_cache()
    return feats


def topk_pairs(feats, names):
    """Retrieval-restricted pairs; exhaustive only for small scenes."""
    n = len(names)
    if n <= EXHAUSTIVE_MAX:
        return [(names[i], names[j]) for i in range(n) for j in range(i + 1, n)]
    emb = np.stack([feats[nm]["gdesc"] for nm in names])
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1)
    pairs = set()
    for i in range(n):
        for j in np.argsort(-sim[i])[:TOPK]:
            a, b = names[i], names[int(j)]
            pairs.add((a, b) if a < b else (b, a))
    return sorted(pairs)


@torch.inference_mode()
def match(feats, pairs):
    lg = KF.LightGlueMatcher("disk").to(DEV).eval()
    out = {}
    for a, b in pairs:
        fa, fb = feats[a], feats[b]
        _, idx = lg(fa["desc"], fb["desc"],
                    KF.laf_from_center_scale_ori(fa["kps"][None].float()),
                    KF.laf_from_center_scale_ori(fb["kps"][None].float()),
                    hw1=torch.tensor(fa["hw"]), hw2=torch.tensor(fb["hw"]))
        if len(idx) >= MIN_MATCHES:
            out[(a, b)] = idx.cpu().numpy().astype(np.uint32)
    return out


def reconstruct(paths, feats, matches, work):
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    db_path = work / "db.db"
    if db_path.exists():
        db_path.unlink()
    db = pycolmap.Database.open(str(db_path))
    ids = {}
    for name, f in feats.items():
        h, w = f["hw"]
        cam = pycolmap.Camera(model="SIMPLE_RADIAL", width=w, height=h,
                              params=[1.2 * max(h, w), w / 2, h / 2, 0.0])
        cid = db.write_camera(cam, use_camera_id=False)
        iid = db.write_image(pycolmap.Image(name=name, camera_id=cid), use_image_id=False)
        ids[name] = iid
        db.write_keypoints(iid, f["kps"].cpu().numpy().astype(np.float32))
    pairs_txt = work / "pairs.txt"
    with open(pairs_txt, "w") as fh:
        for (a, b), m in matches.items():
            db.write_matches(ids[a], ids[b], m); fh.write(f"{a} {b}\n")
    db.close()
    # tuned geometric verification: drop 0-inlier (repeated-structure) pairs + more
    # RANSAC sampling in few-inlier scenes. The one confirmed local win over defaults.
    tvg = pycolmap.TwoViewGeometryOptions()
    tvg.min_num_inliers = 20
    tvg.ransac.max_error = 4.0
    tvg.ransac.confidence = 0.99999
    tvg.ransac.max_num_trials = 100000
    pycolmap.verify_matches(str(db_path), str(pairs_txt), tvg)
    opts = pycolmap.IncrementalPipelineOptions(); opts.ba_refine_focal_length = True
    img_dir = paths[0].parent
    maps = pycolmap.incremental_mapping(str(db_path), str(img_dir), str(work / "sparse"), opts)
    rows = {}
    for midx, rec in maps.items():
        for _, img in rec.images.items():
            if img.has_pose and img.name not in rows:
                pose = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
                rows[img.name] = (f"cluster{midx}", pose.rotation.matrix(),
                                  np.asarray(pose.translation))
    return rows


def _process(paths, ds):
    feats = extract(paths)
    names = [p.name for p in paths]
    pairs = topk_pairs(feats, names)
    print(f"{ds}: {len(paths)} imgs, {len(pairs)} pairs "
          f"(exhaustive={len(names)*(len(names)-1)//2})", flush=True)
    m = match(feats, pairs)
    return reconstruct(paths, feats, m, f"/kaggle/tmp/{ds}")


def write_submission(sub, preds):
    """(Re)write submission.csv from whatever predictions we have so far, so a kill
    at the time limit still leaves a valid, partially-filled file."""
    for i, r in sub.iterrows():
        p = preds.get(r["dataset"], {}).get(r["image"])
        if p:
            sub.at[i, "scene"] = p[0]
            sub.at[i, "rotation_matrix"] = ";".join(f"{v:.9f}" for v in p[1].ravel())
            sub.at[i, "translation_vector"] = ";".join(f"{v:.9f}" for v in p[2])
    sub.to_csv("submission.csv", index=False)


def main():
    sub = pd.read_csv(COMP / "sample_submission.csv")
    preds = {}
    write_submission(sub, preds)  # always leave a valid (baseline) file up front

    # dataset -> image paths; process SMALL datasets first so we bank easy wins
    # before a huge scene can eat the remaining time budget.
    ds_paths = {}
    for ds in sorted(sub.dataset.unique()):
        ds_dir = COMP / "test" / ds
        if not ds_dir.exists():
            continue
        paths = sorted(p for p in ds_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        if len(paths) >= 2:
            ds_paths[ds] = paths
    order = sorted(ds_paths, key=lambda d: len(ds_paths[d]))

    for ds in order:
        if remaining() < 300:  # <5 min left: stop launching new work, keep the file
            print(f"time budget exhausted ({elapsed():.0f}s) -> skipping {ds}", flush=True)
            continue
        try:
            preds[ds] = _process(ds_paths[ds], ds)
            print(f"{ds}: -> {len(preds[ds])} registered "
                  f"[{elapsed():.0f}s elapsed, {remaining():.0f}s left]", flush=True)
        except Exception as e:
            print(f"{ds}: FAILED {str(e)[:120]}", flush=True)
            preds[ds] = {}
        write_submission(sub, preds)  # flush after EVERY dataset

    write_submission(sub, preds)
    print("wrote submission.csv", len(sub), "rows |", elapsed(), "s total", flush=True)


main()
