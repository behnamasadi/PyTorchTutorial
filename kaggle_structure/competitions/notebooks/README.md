# Educational Medical Notebooks — for Kaggle community (notebook) medals

Four polished, **explainable** medical-imaging notebooks built to publish publicly and earn
**upvotes → notebook medals** (Bronze/Silver/Gold). Each is beginner-friendly, well-commented,
and uses explainable AI (Grad-CAM / segmentation overlays). All trained on the local RTX 3090.

## Status — all trained, figures + models saved ✅

| # | Notebook | Dataset (downloads) | Task | Result | Explain | figs |
|---|---|---|---|---|---|---|
| 1 | `chest_xray/` | `paultimothymooney/chest-xray-pneumonia` (640k) | binary cls | **acc 0.83 · AUC 0.95** | Grad-CAM | eda, loss, cm, roc, gradcam |
| 2 | `brain_tumor/` | `sartajbhuvaji/brain-tumor-classification-mri` (100k) | 4-class cls | **acc 0.79** | Grad-CAM | eda, loss, cm, gradcam |
| 3 | `lgg_segmentation/` | `mateuszbuda/lgg-mri-segmentation` | **segmentation** | **Dice 0.87** | GT-vs-pred overlays | eda, dice, overlays |
| 4 | `skin_ham10000/` | `kmader/skin-cancer-mnist-ham10000` (276k) | 7-class cls | **acc 0.84 · bal-acc 0.78** | Grad-CAM | eda, cm, gradcam |
| 5 | `retinal_oct/` | `paultimothymooney/kermany2018` (eyes/OCT) | 4-class cls | **acc 0.98** | Grad-CAM | eda, cm, gradcam |
| 6 | `breast_ultrasound/` | `aryashah2k/breast-ultrasound-images-dataset` | **segmentation** | **Dice 0.69** | GT/pred overlays | eda, dice, overlays |

Each folder has: `*.py` (clean training+viz engine), `figs/` (all figures + `model.pt`), `run.log`.
Chest X-ray also has the **publish-ready notebook**: `pneumonia_efficientnet_gradcam.ipynb`.

## Why these get upvotes
1. **High-traffic datasets** (chest X-ray 640k, skin 276k) → large audience.
2. **Explainable AI** (Grad-CAM / overlays) — a top upvote driver; most notebooks skip it.
3. **Educational**: clean EDA, class-imbalance handling, transfer learning, honest evaluation.
4. **Covers both** classification *and* segmentation.

## The shared recipe (same skeleton in each)
`safe_device()` (P100-safe) → EDA → `timm` EfficientNet / `smp` U-Net → mixed-precision training →
evaluation (accuracy/AUC/confusion-matrix/ROC, or Dice) → **Grad-CAM / mask overlays** → conclusion.
Class imbalance handled with an inverse-frequency weighted loss (chest, skin).

## To publish (do together on review)
For each: create a Kaggle Notebook, attach the dataset, paste the notebook, **Save & Run All**
(so outputs render), then **make public** + a clear title/thumbnail. Grad-CAM figure = the thumbnail.
Datasets are public (no competition/join needed). On Kaggle, GPU is T4/P100 — the `safe_device()`
fallback handles the P100/torch issue (see `../image_matching_2026` notes); training is light enough
for CPU fallback if needed.

## Publish-ready notebook status
- ✅ `chest_xray/pneumonia_efficientnet_gradcam.ipynb` — full markdown + code, 17 cells.
- ⏳ brain_tumor / lgg_segmentation / skin — engines + figures done; `.ipynb` assembly is a quick
  next step (same `build_notebook.py` pattern), to do on review.

## Suggested publish order (by upvote potential)
1. **Chest X-ray** (biggest audience 640k, cleanest, AUC 0.95) — flagship
2. **Retinal OCT** (acc 0.98 — clean, impressive headline; eyes domain)
3. **LGG segmentation** (Dice 0.87; segmentation + overlays stand out)
4. **Skin HAM10000** (imbalance lesson, big audience)
5. **Brain tumor MRI**

Space them out (1–2/day) with distinct titles; a strong Grad-CAM/overlay thumbnail drives clicks.
