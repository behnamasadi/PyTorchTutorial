# A ⭐ Image Matching Challenge 2026

**Kaggle:** `image-matching-challenge-2025-ongoing`. **Type:** ⚠️ **Community (reward: Kudos — NO
MEDALS)**. The medal-bearing IMC editions (2023/2024/2025) are *Research* and all closed.
**Workshop:** CVPR 2026, Image Matching: Local Features & Beyond.

> **Role in the plan:** *practice / edge project*, not a medal path. Leverages the COLMAP/SLAM
> skillset, has a live leaderboard, and makes a strong solution writeup — but for an actual medal
> see `../biohub_cell_tracking/` (the only live Research/medal fit in our domains).

## The task

Given a set of images from one or more scenes (scenes are **not** pre-segmented — you cluster them
yourself), estimate each image's **6-DoF camera pose** (rotation + translation). This is
structure-from-motion / relative-pose estimation — directly in the COLMAP/SLAM wheelhouse.

**Metric:** mean Average Accuracy (**mAA**) of registered cameras — fraction of images whose
estimated pose is within rotation/translation thresholds, averaged over thresholds and scenes.

## Pipeline (what actually wins)

```
images ──▶ (1) scene clustering        global descriptors (DINOv2 / timm), cluster by covisibility
       ──▶ (2) pair generation         retrieval top-k + exhaustive for small scenes
       ──▶ (3) local features+matching  ALIKED / SuperPoint / DISK + LightGlue,  or LoFTR/ELoFTR (detector-free)
       ──▶ (4) geometric verification   RANSAC (fundamental/essential), keep inliers
       ──▶ (5) incremental SfM          pycolmap.match_exhaustive + incremental_mapping → camera poses
       ──▶ (6) submission               rotation_matrix + translation_vector per image, per dataset
```

timm/DINOv2 earns its keep in **(1)–(2)** (retrieval embeddings). The core of the score is the
matcher + SfM quality in **(3)–(5)**. Multi-matcher ensembling (ALIKED+LightGlue **and** LoFTR,
concatenate matches before RANSAC) is the standard top-tier trick.

## Why this is our best medal shot

You already run COLMAP (`magic-inspection-colmap`) and SLAM — the reconstruction engineering that
sinks most Kagglers (registration failures, drift, scene splits) is your day job. That is a real edge.

## Code-competition submission workflow

1. Develop + tune this pipeline locally on the RTX 3090 against the train scenes.
2. Upload matcher weights + wheels (`lightglue`, `pycolmap`, `kornia`) as a Kaggle **Dataset**
   (the submission notebook has **no internet**).
3. Submit an inference-only notebook that imports from that dataset and writes `submission.csv`.

## Status / next steps (needs token + join)

- [ ] Join competition, accept rules, confirm live slug + exact submission schema
- [ ] `kaggle competitions download` train/test scenes → inspect format
- [ ] Wire `pipeline.py` stages to the real data loader
- [ ] Baseline: ALIKED+LightGlue → pycolmap, score train mAA locally
- [ ] Add LoFTR ensemble + retrieval pair-gen, iterate
- [ ] Package offline inference notebook
