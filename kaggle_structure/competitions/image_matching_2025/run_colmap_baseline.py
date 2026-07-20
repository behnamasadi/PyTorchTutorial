"""End-to-end COLMAP baseline for one IMC scene — runs on GPU (SiftGPU).

    python run_colmap_baseline.py <scene_dir> <work_dir>

Reports registered images + extracts (R, t) per image. This is the working
reference reconstruction; the deep-matching upgrade (DISK/ALIKED+LightGlue) swaps
the feature/match stage for the kornia GPU matchers.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap


def main(scene_dir: str, work_dir: str):
    scene = Path(scene_dir)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    db_path = work / "database.db"
    if db_path.exists():
        db_path.unlink()

    imgs = sorted(p.name for p in scene.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    print(f"[colmap] scene={scene.name}  images={len(imgs)}")

    t0 = time.time()
    pycolmap.extract_features(db_path, scene)
    print(f"[colmap] features done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    pycolmap.match_exhaustive(db_path)
    print(f"[colmap] matching done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    maps = pycolmap.incremental_mapping(db_path, scene, work / "sparse")
    print(f"[colmap] mapping done ({time.time()-t0:.1f}s)  models={len(maps)}")

    total_reg = 0
    for idx, rec in maps.items():
        n = rec.num_reg_images()
        total_reg += n
        print(f"  model {idx}: registered {n}/{len(imgs)} images")
        for img_id, img in list(rec.images.items())[:2]:
            pose = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
            R = pose.rotation.matrix()
            t = pose.translation
            print(f"    {img.name}: R{np.round(R,3).tolist()}  t={np.round(t,3).tolist()}")
    print(f"[colmap] TOTAL registered {total_reg}/{len(imgs)} "
          f"({total_reg/max(len(imgs),1):.0%})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
