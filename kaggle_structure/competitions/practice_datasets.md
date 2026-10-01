# Practice datasets — medical / vision quick-wins

Real Kaggle **datasets** (not competitions → no medals, but perfect for portfolio + skill-building)
across the Section-6 categories, chosen for popularity + usability (= clean, ready to train fast).
Download with `kaggle datasets download -d <id>` or `kagglehub.dataset_download("<id>")`.

**All the classification ones reuse our existing template** — `competitions/common/lit_timm.py`
(Lightning + timm) + `gradcam.py`. Segmentation ones use `segmentation_models_pytorch` (2D) or
MONAI (3D). Winning recipe everywhere: pretrained backbone → albumentations aug → k-fold → TTA → small ensemble.

## Classification (→ `common/` Lightning+timm+Grad-CAM template)

| # | Dataset id | Category | Task | Model | Metric |
|---|---|---|---|---|---|
| 1 | `sartajbhuvaji/brain-tumor-classification-mri` | MRI/brain | 4-class | EfficientNetV2-S / ConvNeXt-T | accuracy |
| 2 | `paultimothymooney/chest-xray-pneumonia` | chest X-ray | binary | DenseNet-121 / EffNet-B0 | AUC/F1 |
| 3 | `tawsifurrahman/tuberculosis-tb-chest-xray-dataset` | chest X-ray | binary (TB) | EfficientNet-B0 + Grad-CAM | AUC |
| 4 | `paultimothymooney/kermany2018` | eyes/OCT | 4-class | EfficientNet-B3 | accuracy |
| 5 | `kmader/skin-cancer-mnist-ham10000` | skin/lesion | 7-class (imbalanced) | EfficientNet + focal loss / class-weights | balanced-acc |
| 6 | `paultimothymooney/breast-histopathology-images` | histopathology | binary patch (IDC) | ResNet-50 / EffNet-B0, heavy TTA | AUC |
| 7 | `andrewmvd/leukemia-classification` | blood smear | binary (ALL) | EfficientNet-B0 | AUC |
| 8 | `sshikamaru/glaucoma-detection` | eyes | binary | EfficientNet + Grad-CAM | AUC |

## Segmentation (→ `segmentation_models_pytorch` 2D / MONAI 3D)

| # | Dataset id | Category | Task | Model | Metric |
|---|---|---|---|---|---|
| 9 | `mateuszbuda/lgg-mri-segmentation` | MRI/brain | 2D seg (FLAIR) | U-Net++ w/ EffNet encoder (smp) | Dice |
| 10 | `aryashah2k/breast-ultrasound-images-dataset` (BUSI) | ultrasound/breast | 2D seg + class | U-Net + aux classifier | Dice |
| 11 | `dankok/kvasir-seg` / `orvile/cvc-clinicdb` | endoscopy | 2D polyp seg | SegFormer / U-Net++ | Dice/IoU |
| 12 | `andrewmvd/liver-tumor-segmentation` (LiTS) | CT/liver | 3D seg | MONAI SegResNet / 3D U-Net | Dice/HD95 |
| 13 | `andrewmvd/covid19-ct-scans` | CT/lungs | 3D seg + class | MONAI UNet + sliding-window | Dice |

## Recommended pipeline per task type

- **2D classification:** `TimmClassifier` (our template) → EfficientNetV2-S, 5-fold, CLAHE/preproc,
  albumentations, TTA, Grad-CAM for explainability. ~1–2h to a strong result on the 3090.
- **2D segmentation:** `smp.UnetPlusPlus(encoder_name="timm-efficientnet-b4")`, Dice+BCE loss,
  albumentations (flips/elastic), TTA, threshold search on OOF.
- **3D segmentation:** MONAI `SegResNet`/`UNet` + `CacheDataset` + sliding-window inference +
  `DiceCELoss`; napari for 3D inspection. Fits 24 GB with patch-based training.

## Suggested order (fastest → most impressive)
1. Chest X-ray pneumonia (#2) — biggest, cleanest, ~30 min to >0.95 AUC. Warm-up.
2. Brain tumor MRI classification (#1) + LGG segmentation (#9) — same brain domain, class + seg combo.
3. Skin HAM10000 (#5) — teaches class-imbalance handling.
4. LiTS liver (#12) — your first real 3D MONAI segmentation.
