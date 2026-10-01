# Notebook submission status & quick-publish guide

Last updated: 2026-07-21. Owner Kaggle user: **asadibehnam**.

All notebooks are **GPU-forced** (install `torch==2.5.1+cu121` → `assert cuda` → auto-find data path).
They CANNOT fall back to CPU. Each folder has the polished `.ipynb` + a ready `kernel-metadata.json`.

## ✅ PUBLISHED (live, public, GPU-run) — 10

| Notebook | Folder | Kaggle URL (code/asadibehnam/…) | Result |
|---|---|---|---|
| Pneumonia — EfficientNet + Grad-CAM | `chest_xray/` | `95-auc-pneumonia-detector-grad-cam` | AUC ~0.95 |
| Brain Tumor Segmentation — U-Net | `lgg_segmentation/` | `0-88-dice-brain-tumor-mri-segmentation-u-net` | Dice ~0.88 |
| Retinal OCT — EfficientNet + Grad-CAM | `retinal_oct/` | `98-5-retinal-oct-classifier-grad-cam` | acc **0.9845** |
| Skin Lesion HAM10000 — EfficientNet + Grad-CAM | `skin_ham10000/` | `leak-free-skin-cancer-classifier-ham10000` | acc **0.763** / bal **0.678** (leak-free) |
| **BRISC 2025** Brain Tumor MRI — EfficientNet + Grad-CAM | `brisc_brain_2025/` | `98-brain-tumor-mri-classifier-grad-cam` | acc **0.982** / bal **0.984** (2025 dataset) |
| Retinal Vessel Segmentation — U-Net++ | `retina_vessel/` | `retinal-vessels-81-dice-with-u-net` | Dice **0.814** (2023 dataset) |
| 🩺 Breast Tumor Segmentation: A U-Net++ Upgrade | `breast_ultrasound/` | `breast-tumor-segmentation-a-u-net-upgrade` | **0.778 Dice** (v3, beats published ~0.75) |
| 🔬 Does Your Cancer Classifier Actually Work? (BreakHis 400x) | `breakhis_histopathology/` | `does-your-cancer-classifier-actually-work` | **patient-level 0.882 acc / 0.865 AUC** (2025 dataset, leak-free + voting) |
| 🔬 99% Lung & Colon Cancer Detection + Grad-CAM (LC25000) | `lung_colon_histopath/` | `99-lung-colon-cancer-detection-grad-cam` | **0.9994 accuracy** (5-class) |
| Can AI Spot Liver Cancer in an Ultrasound? | `liver_ultrasound/` | `can-ai-spot-liver-cancer-in-an-ultrasound` | seg Dice **0.647** + 3-class acc **0.728** (2025 dataset, seg+cls+Grad-CAM) |

Bottom 4 published **2026-07-21**. All GPU-verified, error-free, eye-catching titles.

## 🗺️ NEXT BATCH — vetted, not yet built (2026-07-21 roadmap)

Priority order (reuse the existing EfficientNet+Grad-CAM / U-Net++ engines):

| P | Idea | Dataset ref | Task | Why |
|---|---|---|---|---|
| 1 | GastroVision GI endoscopy | `orvile/gastrovision-gastrointestinal-disease-detection` | cls | **new modality** (endoscopy), 2025, usability 1.0 |
| 2 | KiTS19 Kidney Tumor seg | `orvile/kits19-png-zipped` | seg | new organ, 2025, PNG; 6.7 GB |
| 3 | Depth Anything v2 demo | *(pretrained, no dataset)* | monocular depth | plays to SfM/photogrammetry expertise, stunning viz, no training |
| 4 | Liver US **mass seg only, harder losses** | `orvile/annotated-ultrasound-liver-images-dataset` | seg | already have the data; Tversky/boundary loss to push Dice > 0.65 |
| — | Cardiac ACDC seg | `samdazel/automated-cardiac-diagnosis-challenge-miccai17` | seg | new organ (heart) but 3D/complex |

**Dropped:** old Brain-Tumor-MRI (`sartajbhuvaji`, 8-yr-old dataset) — redundant with BRISC 2025 (0.982) and scored worse (0.79). `brain_tumor/` folder can be deleted.

## ⚠️ Publish gotchas — hard-won 2026-07-21 (READ before pushing)

- **CREATE (new kernel) needs id == title-slug, plain ASCII, NO emoji, NO leading digit.**
  - Emoji in the title on *create* → `Kernel push error: Notebook not found`. Create with a plain ASCII title.
  - Leading-digit id (`99-lung-...`) → same "Notebook not found". Start ids with a letter.
  - `99`/`grad-cam` inside the id also tripped it once; safest id = plain words + hyphens.
- **UPDATE (kernel already exists) allows anything** — emoji, `&`, `%`, mismatched slug. You get a *warning*
  ("title does not resolve to the specified id"), the push still succeeds, and Kaggle **re-slugs the public
  URL to the new title**. So the workflow that works:
  1. Create with a clean ASCII title/id (goes public/private, runs).
  2. Push again (update) with the 🔬-emoji clickbait title → URL re-slugs to the catchy version.
- **Re-slug drift:** after an update re-slugs the URL, the OLD ref 404s. Before the *next* push, set
  `id` in metadata to the CURRENT ref (from `kaggle kernels list --user asadibehnam`) or you'll create a
  duplicate. (Bit us on breast: `…u-net-dice` → `…u-net-wins-6` → `…a-u-net-upgrade`.)
- **`kaggle kernels push` ALWAYS re-runs on GPU** — there is no metadata-only update; flipping `is_private`
  or changing a title costs a full re-run. Push **public-first** (`is_private:false`) for notebooks you're
  confident in to avoid a second run.
- **Max 2 concurrent GPU kernels = HARD REJECT, not a queue.** A 3rd push errors
  (`Maximum batch GPU session count of 2 reached`). Retry-loop every 60s until a slot frees.
- **No hardcoded % in prose/titles.** Baselines shift run-to-run (breast v1 +5.8%, v2 +4.0%, v3 +4.2%),
  so a static "+9%"/"+6%" contradicts the notebook's own printed output. State absolute Dice/acc
  (~0.78) or let the cell compute the delta; keep numbers OUT of static markdown.
- **No em-dash (U+2014) in code cells** → `SyntaxError` on Kaggle. Keep prose out of code cells.
- **Leakage is the recurring headline hook** — patient/lesion-grouped splits (HAM10000, BreakHis) and
  patient-level voting (BreakHis 0.72 patch → 0.882 patient) are what make the honest number credible.

## HOW TO PUBLISH (clean-title-first workflow)

```bash
cd competitions/notebooks/<folder>
# 1. metadata: plain ASCII title, id == its slug, is_private:false, GPU+internet+dataset set
kaggle kernels push                                   # creates + runs public on GPU
kaggle kernels status asadibehnam/<id>                # wait for COMPLETE
kaggle kernels output asadibehnam/<id> -p out         # verify numbers + no Traceback
# 2. (optional) add emoji/clickbait title -> update push re-slugs the URL:
python -c "import json;m=json.load(open('kernel-metadata.json'));m['title']='🔬 Catchy Title';json.dump(m,open('kernel-metadata.json','w'),indent=1)"
kaggle kernels push                                   # re-runs; URL becomes the catchy slug
```

## Medals note (2026-07-21)

Profile `kaggle.com/asadibehnam` shows **tier badges** (Novice/Contributor/…), NOT medals — different systems.
Medals (🥉≥5 / 🥈≥20 / 🥇≥50 votes) count **only non-novice upvotes** (self + novice votes excluded), shown
per-category on the profile cards and as an icon on each notebook. Currently **0 medals** (raw vote counts of
9/7 on the two top older notebooks aren't from enough qualifying users). Eye-catching titles target the 5-vote
Bronze bar.

## Gotchas already handled (baked into every notebook)
- **Kaggle P100 + default torch = broken** → every notebook installs `torch==2.5.1+cu121` first.
- **`grad-cam` / `smp` / `timm` / `albumentations` not preinstalled** → each notebook `!pip install`s them.
- **Data paths nest differently on Kaggle** → notebooks auto-find the data dir (no hardcoded paths).
