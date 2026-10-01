# B — APTOS 2019 Diabetic Retinopathy

**Kaggle:** `aptos2019-blindness-detection` (retinal fundus). **Task:** ordinal 5-class DR grading
(0 No DR → 4 Proliferative). **Metric:** quadratic weighted kappa (QWK).

This is the **reference pipeline** — the reusable Lightning + timm + Grad-CAM skeleton in
`../common/` — and a high-confidence strong result (top solutions reach QWK ≈ 0.90+).

## Approach

- **Preprocessing:** Ben Graham / CLAHE contrast normalization + circular crop of the fundus.
- **Model:** `timm` EfficientNet-B4/B5 (or EfficientNetV2-S), **ordinal regression** head
  (scalar output, MSE loss, round to class) — matches the QWK metric better than plain softmax.
- **Validation:** 5-fold stratified CV; report per-fold QWK.
- **Inference:** TTA (h/v flip) + optimized rounding thresholds (`scipy.optimize` on OOF preds).
- **Explainability:** Grad-CAM overlays (`../common/gradcam.py`) to confirm the model attends to
  microaneurysms / hemorrhages, not borders.

## Run (after token + `kaggle competitions download -c aptos2019-blindness-detection`)

```bash
python train.py --data ./data --backbone tf_efficientnetv2_s.in21k_ft_in1k --folds 5 --epochs 20
```

## Status

- [ ] Download data (needs token + join)
- [ ] Wire `train.py` datamodule to the CSV + image folder
- [ ] Baseline single-fold → confirm QWK, then 5-fold + TTA + threshold opt
- [ ] Grad-CAM figure grid for the writeup
