"""Post-reconstruction SfM-quality metrics — the half `analyze_failures.py` is missing.

Modelled on magic-inspection-colmap's "Manual analysis — the evidence" (B0-playbook):
every metric gets a value + a ✓/⚠/✗ chip from fixed thresholds + what it means, laid
out in data-flow order. Rule of thumb: **read the first ✗ in data-flow order — it is
usually the root cause.** `analyze_failures.py` diagnoses the match graph BEFORE mapping;
this diagnoses the reconstruction AFTER, which is where "registered but geometrically
wrong" hides (fbk_vineyard registers 95 imgs yet scores rot_mAA 0.10 — fragmentation,
not matching).

IMC-specific calibration (differs from the single-scene aerial playbook): an IMC dataset
holds SEVERAL GT scenes, so >1 COLMAP model is EXPECTED, not a fault. The real
fragmentation signal is per-GT-scene coherence — did each GT scene stay inside one model
— which needs train labels; at test time we fall back to dataset-level shape only.

    from sfm_diagnostics import diagnose, evidence_table, root_cause
    diag = diagnose(maps, names, gt_scene={img: scene, ...})   # gt_scene optional (train)
    print(evidence_table(diag, name))
"""
from __future__ import annotations

import numpy as np

_CHIP = {"ok": "✓", "warn": "⚠", "bad": "✗", "na": "·"}


def _band(v, good, warn, higher_better=True):
    if v is None:
        return "na"
    if higher_better:
        return "ok" if v >= good else "warn" if v >= warn else "bad"
    return "ok" if v <= good else "warn" if v <= warn else "bad"


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _reg_images(rec):
    return [im for im in rec.images.values() if getattr(im, "has_pose", False)]


def _model_stats(rec):
    reg = _reg_images(rec)
    pts = rec.points3D
    reproj = _safe(rec.compute_mean_reprojection_error)
    if reproj is None:
        errs = [e for e in (_safe(lambda p=p: float(p.error)) for p in pts.values())
                if e is not None and e >= 0]
        reproj = float(np.mean(errs)) if errs else None
    track = _safe(rec.compute_mean_track_length)
    if track is None:
        lens = [L for L in (_safe(lambda p=p: p.track.length()) for p in pts.values())
                if L is not None]
        track = float(np.mean(lens)) if lens else None
    obs_img = _safe(rec.compute_mean_observations_per_reg_image)
    if obs_img is None and reg:
        n_obs = _safe(rec.compute_num_observations)
        obs_img = n_obs / len(reg) if n_obs is not None else None
    return {"reg_names": [im.name for im in reg], "reg": len(reg),
            "points3D": len(pts), "mean_track_length": track,
            "mean_reproj_px": reproj, "obs_per_image": obs_img}


def _gt_coherence(models, gt_scene):
    """Per-GT-scene coherence: fraction of each GT scene's registered images that
    landed in ONE model, and how many models it spilled across. Returns the WORST
    (lowest-coherence) scene — the true IMC fragmentation signal."""
    # image -> model index
    img2model = {}
    for mi, m in enumerate(models):
        for nm in m["reg_names"]:
            img2model[nm] = mi
    by_scene = {}
    for img, sc in gt_scene.items():
        mi = img2model.get(img)
        if mi is not None:
            by_scene.setdefault(sc, []).append(mi)
    worst = None
    for sc, mis in by_scene.items():
        if not mis:
            continue
        counts = np.bincount(mis)
        coherence = counts.max() / counts.sum()   # 1.0 = scene fully in one model
        spanned = int((counts > 0).sum())
        rec = {"scene": sc, "coherence": float(coherence), "models_spanned": spanned,
               "n_reg": int(counts.sum())}
        if worst is None or coherence < worst["coherence"]:
            worst = rec
    return worst


def diagnose(maps, names, gt_scene=None):
    """Aggregate diagnostics for one dataset. `names` = all image names in the
    dataset; `gt_scene` = {image_name: gt_scene_label} when known (train)."""
    models = list(maps.values()) if hasattr(maps, "values") else list(maps)
    per = [s for s in (_model_stats(m) for m in models) if s["reg"] > 0]
    n_images = len(names)
    total_reg = sum(p["reg"] for p in per)
    out = {"n_images": n_images, "n_models": len(per), "reg": total_reg,
           "reg_rate": total_reg / max(n_images, 1),
           "largest": max(per, key=lambda p: p["reg"]) if per else None,
           "worst_scene": None, "n_gt_scenes": None}
    if gt_scene:
        out["n_gt_scenes"] = len(set(gt_scene.values()))
        out["worst_scene"] = _gt_coherence(per, gt_scene) if per else None
    # strip bulky name lists before returning/caching
    for p in per:
        p.pop("reg_names", None)
    out["models"] = per
    return out


def _rows(diag):
    """Evidence rows in DATA-FLOW order: (metric, value, status, meaning)."""
    lm = diag["largest"] or {}
    rows = [
        ("Registration coverage", f"{diag['reg']}/{diag['n_images']} ({diag['reg_rate']:.0%})",
         _band(diag["reg_rate"], 0.9, 0.6),
         "≥90% good · 60–90% worry · <60% pairing/verification too sparse"),
    ]
    # IMC fragmentation: prefer per-GT-scene coherence (train), else dataset shape
    ws = diag.get("worst_scene")
    if ws is not None:
        rows.append((
            f"Worst-scene coherence ({ws['scene']})",
            f"{ws['coherence']:.2f} in {ws['models_spanned']} model(s)",
            _band(ws["coherence"], 0.9, 0.6),
            "1.0 = GT scene kept in one model · <0.6 = scene split → mAA collapses"))
    elif diag["n_gt_scenes"] is None and diag["n_models"]:
        rows.append(("Models produced", str(diag["n_models"]), "na",
                     "multi-scene dataset → >1 expected; can't judge split without labels"))
    rows += [
        ("Mean track length", None if lm.get("mean_track_length") is None
         else f"{lm['mean_track_length']:.1f} obs/pt",
         _band(lm.get("mean_track_length"), 4.0, 3.0),
         "≥4 solid multi-view · 3–4 thin · <3 poorly-constrained triangulation"),
        ("Mean reproj error", None if lm.get("mean_reproj_px") is None
         else f"{lm['mean_reproj_px']:.2f} px",
         _band(lm.get("mean_reproj_px"), 1.0, 1.5, higher_better=False),
         "<1.0 good — but read next to coverage (low error on few imgs is a trap)"),
        ("3D points (largest model)", None if lm.get("points3D") is None
         else f"{lm['points3D']:,}",
         _band(lm.get("points3D"), 2000, 300),
         "thousands good · hundreds thin · <~200 bad"),
    ]
    return rows


def root_cause(diag):
    """The first ✗ (else first ⚠) in data-flow order — the platform's rule."""
    if not diag or diag["n_models"] == 0:
        return "✗ no reconstruction (matching/verification produced no usable model)"
    bad = [(m, v) for m, v, s, _ in _rows(diag) if s == "bad"]
    if bad:
        return f"✗ {bad[0][0]} = {bad[0][1]}"
    warn = [(m, v) for m, v, s, _ in _rows(diag) if s == "warn"]
    if warn:
        return f"⚠ {warn[0][0]} = {warn[0][1]}"
    return "✓ SfM-healthy — if mAA still low, cause is symmetry/rotation-prior, not SfM"


def evidence_table(diag, name=""):
    if not diag or diag["n_models"] == 0:
        return f"[diag] {name}: {root_cause(diag)}"
    lines = [f"[diag] {name}: {root_cause(diag)}  "
             f"(models={diag['n_models']}"
             + (f", gt_scenes={diag['n_gt_scenes']}" if diag['n_gt_scenes'] else "") + ")"]
    for metric, value, status, meaning in _rows(diag):
        lines.append(f"    {_CHIP.get(status,'·')} {metric:<28} "
                     f"{(value or '—'):<22} {meaning}")
    return "\n".join(lines)
