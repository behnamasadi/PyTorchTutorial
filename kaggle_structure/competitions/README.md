# Kaggle Medal Competitions

Working code for the three active targets (see Section 8 of `../index.ipynb`).

```
competitions/
├── common/                     # reusable Lightning + timm + Grad-CAM building blocks (shared by B & C)
│   ├── lit_timm.py             # TimmClassifier LightningModule (classification / ordinal)
│   ├── gradcam.py              # Grad-CAM / saliency overlays
│   └── requirements.txt
├── image_matching_2026/        # A ⭐  SfM: retrieval -> matching -> pycolmap poses
├── aptos_dr/                   # B     diabetic retinopathy, ordinal 5-class
└── biohub_cell_tracking/       # C     3D cell segmentation + tracking
```

## Before anything runs — clear the gates

1. **Kaggle token** — download from <https://www.kaggle.com/settings> → *Create New Token*, then:
   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
2. **Join each competition** on the website and **accept its rules** (cannot be automated).
3. **Code competitions** (IMC, RSNA): the submission is an *offline notebook* that runs on Kaggle's
   servers with no internet. Workflow: develop here → train weights on the local RTX 3090 → upload
   weights + matcher wheels as a Kaggle *Dataset* → submit an inference-only notebook.

## Environment

Training runs inside the official Kaggle image (has torch/CUDA/timm/MONAI preinstalled):

```bash
docker run --rm --gpus all -it \
  -v "$PWD":/work -w /work \
  -v ~/.kaggle:/root/.kaggle:ro \
  gcr.io/kaggle-gpu-images/python:latest /bin/bash
```

Or a local venv:  `pip install -r common/requirements.txt`
