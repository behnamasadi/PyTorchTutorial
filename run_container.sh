#!/bin/bash
# Script to run the Docker container with GPU support
# Override the default entrypoint (sleep infinity) to run bash instead
#
# Volume Mount Strategy:
# - /workspace/projects  → projects from Dockerfile image (persistent)
# - /workspace/host      → your host home directory (mounted)
#
# This allows both the image's projects and your host files to coexist.

echo "Starting container with GPU support..."

# Override entrypoint to run bash instead of sleep infinity
# Mount host home directory to /workspace/host (not /workspace) to preserve image projects
docker run -it --gpus all \
  --entrypoint bash \
  -e KAGGLE_USERNAME=$KAGGLE_USERNAME \
  -e KAGGLE_KEY=$KAGGLE_KEY \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e HOME=/workspace \
  -v $HOME:/workspace/host \
  ghcr.io/behnamasadi/kaggle-projects:latest

# Alternative if --gpus all doesn't work, try:
# docker run -it --runtime=nvidia \
#   --entrypoint bash \
#   -e KAGGLE_USERNAME=$KAGGLE_USERNAME \
#   -e KAGGLE_KEY=$KAGGLE_KEY \
#   -e WANDB_API_KEY=$WANDB_API_KEY \
#   -e HOME=/workspace \
#   -v $HOME:/workspace/host \
#   ghcr.io/behnamasadi/kaggle-projects:latest

