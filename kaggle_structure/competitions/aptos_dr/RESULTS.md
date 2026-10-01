# APTOS DR (B) — results

Trained on the **public APTOS-2019 mirror** (`sovitrath/diabetic-retinopathy-224x224-2019-data`,
3,662 labeled fundus images) because the competition download is gated on joining. Same
Lightning + timm + Grad-CAM pipeline that would run on the real competition data.

## Runs

| Config | val QWK |
|---|---|
| EfficientNetV2-S ordinal, single fold, 8 ep, CLAHE | 0.8135 |
| **+ fundus crop + circular mask + 5-fold ensemble, 12 ep** | **0.8355 ± 0.011** (folds 0.844/0.852/0.824/0.829/0.829) |

## Grad-CAM findings
- v1 revealed **shortcut learning** — the model attended to the black image corners as much as the retina.
- Fix (tight fundus crop + circular mask) shifted attention onto the **optic disc + distributed
  hemorrhages** (clinically sensible), esp. on severe (gt=4) cases. Faint corner CAM residue remains
  but corners are zeroed → it's a CAM edge artifact, not model reliance.

## Ceiling on this data
Mirror images are 224px → resolution-capped. Top APTOS solutions (~0.90+) use the **original
high-res** images + TTA + backbone-diverse ensembles. This run proves the pipeline; the real
competition data (higher res) + those tricks would close the gap.

## Reproduce
    python train_public.py --data <mirror_path> --epochs 12 --folds 5 --size 256
