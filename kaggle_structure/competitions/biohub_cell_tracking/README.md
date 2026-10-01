# C — Biohub: Cell Tracking During Development  (live medal)

**Kaggle:** `biohub-cell-tracking-during-development` (opened 2026-06-29, ~2 months left).
**Task:** detect + track zebrafish cells through 3D space and time. **Metric:** SEG/tracking
(Jaccard / CHOTA-style). *Alternative for slot C:* `rsna-intracranial-aneurysm-detection`.

## Approach

```
3D volumes ──▶ (1) detection/segmentation   StarDist-3D or 3D U-Net (MONAI SegResNet), per timepoint
           ──▶ (2) linking                   nearest-neighbor / graph tracker (overlap + motion) across t
           ──▶ (3) track post-proc           gap closing, division handling, ID stitching
```

- **Segmentation:** MONAI `SegResNet` / `UNet` (3D) or StarDist-3D for star-convex nuclei; sliding-window
  inference to fit 24 GB. Dice + boundary loss.
- **Tracking:** start from the official nearest-neighbor baseline, then upgrade to overlap/graph linking
  (e.g. Hungarian on IoU + predicted motion); handle cell **divisions** (1→2 tracks).
- **Viz:** `napari` for 3D+t inspection of masks and tracks.

## References

- Official metrics + baseline: <https://github.com/royerlab/kaggle-cell-tracking-competition>
- Classical baseline notebook + nearest-neighbor starter (linked from the competition).

## Status (needs token + join)

- [ ] Join competition, accept rules, confirm metric + submission format + data size
- [ ] Download → inspect 3D+t volume format
- [ ] 3D segmentation baseline (MONAI) → per-frame masks
- [ ] Nearest-neighbor tracker → first submission
- [ ] Upgrade linking (graph/overlap) + division handling, iterate
