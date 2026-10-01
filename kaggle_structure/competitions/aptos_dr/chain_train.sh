#!/bin/bash
# wait for the IMC RoMa head-to-head to release the GPU, then train APTOS
while pgrep -f run_roma >/dev/null 2>&1; do sleep 15; done
echo "[chain] RoMa done, GPU free -> starting APTOS training $(date)"
docker run --rm --gpus all --shm-size=8g   -v /home/behnam/workspace/PyTorchTutorial/kaggle_structure/competitions:/comp   -v /home/behnam/.cache/kagglehub/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data/versions/4:/data:ro   -w /comp/aptos_dr   gcr.io/kaggle-gpu-images/python:latest   bash -c "pip install -q grad-cam >/dev/null 2>&1; python train_public.py --data /data --epochs 8 --folds 1"
echo "[chain] APTOS training done $(date)"
