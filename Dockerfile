# Use the same base image Kaggle uses for GPU notebooks
# This ensures your RunPod environment matches Kaggle notebooks exactly
# See: https://github.com/kaggle/docker-python
FROM gcr.io/kaggle-gpu-images/python

ENV DEBIAN_FRONTEND=noninteractive

# Optional: only add extra system deps you actually need
# Most common ML/data packages (torch, pandas, sklearn, etc.) are already installed
# See: https://github.com/Kaggle/docker-python/blob/main/kaggle_requirements.txt
RUN apt-get update && apt-get install -y \
    git wget unzip ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to /workspace
WORKDIR /workspace

# Universal Python dependencies (optional - only if you have common packages across all projects)
# Most common packages (torch, pandas, numpy, sklearn, matplotlib, etc.) are pre-installed in Kaggle image
# Only install packages NOT already in the Kaggle image that you use across multiple projects
# For project-specific packages, install them per-project or mount them at runtime
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy projects directory (contains all your Kaggle challenge projects)
# When running with -v $HOME:/workspace/host, your host home directory is accessible at /workspace/host
# This allows you to work with multiple projects in the same image
COPY projects/ projects/

ENTRYPOINT ["sleep", "infinity"]
