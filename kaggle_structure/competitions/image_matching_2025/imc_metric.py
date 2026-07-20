"""Local IMC scoring: registration rate + rotation/translation mAA.

Metric functions (rotation_error_deg, translation_direction_error_deg, map_aa,
umeyama_sim3) are adapted from magic-inspection-colmap
`scripts/pipeline/holdout_localization.py` — the same ETH3D/IMC mAA formulation
used in that pipeline. Reused here to build a local validation harness for the
Kaggle Image Matching Challenge (reconstruction is up to a similarity transform,
so we Umeyama-align predicted camera centers to GT before scoring).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROT_THRESHOLDS_DEG = [1.0, 2.0, 3.0, 5.0, 10.0]


def rotation_error_deg(r_gt, r_est) -> float:
    a = np.asarray(r_gt, np.float64).reshape(3, 3)
    b = np.asarray(r_est, np.float64).reshape(3, 3)
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def map_aa(errors_deg, thresholds) -> float:
    """Mean over thresholds of the fraction of errors below each threshold."""
    n = len(errors_deg)
    if n == 0:
        return 0.0
    return float(np.mean([np.mean([e <= t for e in errors_deg]) for t in thresholds]))


def umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    """Least-squares Sim3 (s, R, t) with dst ≈ s*R@src + t (Umeyama 1991)."""
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    sc, dc = src - src_mean, dst - dst_mean
    cov = dc.T @ sc / sc.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[2, 2] = -1.0
    rot = u @ s_mat @ vt
    var = (sc ** 2).sum(1).mean()
    scale = (d * np.diag(s_mat)).sum() / var if var > 0 else 1.0
    trans = dst_mean - scale * rot @ src_mean
    return float(scale), rot, trans


def _center(R, t):
    """Camera center in world coords from cam_from_world (R, t): C = -R^T t."""
    return -np.asarray(R, np.float64).T @ np.asarray(t, np.float64)


def robust_umeyama(src, dst, iters=200, thr_frac=0.1):
    """RANSAC Umeyama — robust to images split across COLMAP models (mixed frames).

    Non-robust Umeyama averages over all points; if a GT scene's images are in two
    reconstructions with different frames, the fit is garbage and every pose scores
    as wrong. RANSAC finds the majority frame. Returns (s, R, t, inlier_mask).
    """
    n = len(src)
    if n < 3:
        return (*umeyama_sim3(src, dst), np.ones(n, bool))
    scale_ref = np.linalg.norm(dst - dst.mean(0), axis=1).mean() + 1e-9
    best_inl = None
    rng = np.random.RandomState(0)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            s, R, t = umeyama_sim3(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        resid = np.linalg.norm((s * (R @ src.T).T + t) - dst, axis=1)
        inl = resid < thr_frac * scale_ref
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl = inl
    if best_inl is None or best_inl.sum() < 3:
        return (*umeyama_sim3(src, dst), np.ones(n, bool))
    s, R, t = umeyama_sim3(src[best_inl], dst[best_inl])
    return s, R, t, best_inl


def parse_pose(row):
    R = np.array([float(x) for x in row["rotation_matrix"].split(";")]).reshape(3, 3)
    t = np.array([float(x) for x in row["translation_vector"].split(";")])
    return R, t


def score_dataset(preds: dict, labels_csv, thresholds_csv, dataset: str) -> dict:
    """preds: {image_name: (R 3x3, t 3)} cam_from_world for registered images.

    Returns per-scene registration rate + rotation mAA + translation mAA
    (translation at the scene's official thresholds, after Sim3 alignment).
    """
    gt = pd.read_csv(labels_csv)
    gt = gt[gt.dataset == dataset]
    thr = pd.read_csv(thresholds_csv)
    thr = thr[thr.dataset == dataset]
    scene_thr = {r["scene"]: [float(x) for x in r["thresholds"].split(";")]
                 for _, r in thr.iterrows()}

    results = {}
    for scene, g in gt.groupby("scene"):
        if scene == "outliers":
            continue
        names = list(g.image)
        reg = [n for n in names if n in preds]
        rate = len(reg) / max(len(names), 1)
        if len(reg) < 3:
            results[scene] = {"n": len(names), "registered": len(reg),
                              "reg_rate": round(rate, 3), "rot_mAA": 0.0,
                              "trans_mAA": 0.0, "combined_mAA": 0.0}
            continue
        gt_map = {r["image"]: parse_pose(r) for _, r in g.iterrows()}
        C_pred = np.array([_center(*preds[n]) for n in reg])
        C_gt = np.array([_center(*gt_map[n]) for n in reg])
        s, R_align, t_align, _ = robust_umeyama(C_pred, C_gt)

        rot_errs, pos_errs = [], []
        for i, n in enumerate(reg):
            R_p, _ = preds[n]
            R_g, _ = gt_map[n]
            rot_errs.append(rotation_error_deg(R_g, R_p @ R_align.T))
            pos_errs.append(float(np.linalg.norm(s * R_align @ C_pred[i] + t_align - C_gt[i])))
        # unregistered images count as failures (append inf)
        miss = len(names) - len(reg)
        rot_all = rot_errs + [1e9] * miss
        pos_all = pos_errs + [1e9] * miss
        rthr = ROT_THRESHOLDS_DEG
        pthr = scene_thr.get(scene, [0.1])
        # COMBINED (official IMC): a camera counts only if rotation AND translation are
        # BOTH within threshold, averaged over all (rot_thr, trans_thr) pairs. This is
        # the number the Kaggle leaderboard reports — rot_mAA/trans_mAA alone are lenient.
        combined = float(np.mean([
            np.mean([(rot_all[i] <= rt and pos_all[i] <= pt) for i in range(len(rot_all))])
            for rt in rthr for pt in pthr]))
        results[scene] = {
            "n": len(names), "registered": len(reg), "reg_rate": round(rate, 3),
            "rot_mAA": round(map_aa(rot_all, rthr), 4),
            "trans_mAA": round(map_aa(pos_all, pthr), 4),
            "combined_mAA": round(combined, 4),
            "median_rot_err": round(float(np.median(rot_errs)), 2),
        }
    return results
