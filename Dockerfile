FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git wget unzip ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to /workspace
WORKDIR /workspace

# Reinstall PyTorch from pip to get broader CUDA kernel support
# The pip builds include kernels for a wider range of GPU compute capabilities (3.5, 5.0, 6.0, 6.1, 7.0, 7.5, 8.0, 8.6, 8.9, 9.0)
# This ensures compatibility with various GPU architectures
# Using CUDA 12.4 which is widely supported on RunPod and other cloud platforms
RUN pip uninstall -y torch torchvision torchaudio 2>/dev/null || true && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy projects to /workspace/projects (from Dockerfile image)
# When running with -v $HOME:/workspace/host, this directory remains visible
# and your host home directory is accessible at /workspace/host
COPY projects/ projects/

ENTRYPOINT ["sleep", "infinity"]
