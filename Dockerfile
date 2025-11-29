FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git wget unzip ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Reinstall PyTorch from pip to get broader CUDA kernel support
# The pip builds include kernels for a wider range of GPU compute capabilities (3.5, 5.0, 6.0, 6.1, 7.0, 7.5, 8.0, 8.6, 8.9, 9.0)
# This ensures compatibility with various GPU architectures
# Using CUDA 12.4 build which is compatible with CUDA 13.0 runtime (backward compatible)
RUN pip uninstall -y torch torchvision torchaudio 2>/dev/null || true && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY projects/ projects/

ENTRYPOINT ["sleep", "infinity"]
