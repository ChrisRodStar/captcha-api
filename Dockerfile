FROM oven/bun:1-debian
WORKDIR /app

# Install system dependencies for onnxruntime-node (CPU) + healthcheck + npm for CUDA binary installation
RUN apt-get update && apt-get install -y \
    libgomp1 \
    wget \
    gnupg2 \
    curl \
    npm \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA runtime libraries for GPU support (required by onnxruntime-node CUDA provider)
# Add NVIDIA repo with trusted=yes only (no cuda-keyring) to avoid GPG/SHA1 conflict on Debian Trixie
RUN echo 'deb [trusted=yes] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /' > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    cuda-cudart-12-2 \
    cuda-nvrtc-12-2 \
    libcublas-12-2 \
    libcurand-12-2 \
    libcusolver-12-2 \
    libcusparse-12-2 \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

# Note: For GPU support:
# 1. NVIDIA GPU with CUDA 12.x drivers installed on the host (✓ verified working)
# 2. nvidia-container-toolkit installed on the host (✓ verified working)
# 3. runtime: nvidia configured in docker-compose.yml (✓ configured)
# 4. CUDA runtime libraries are installed above (libcublas, libcudnn, etc.)
#    These are required for onnxruntime-node CUDA provider to work

# Install dependencies with Bun (no --frozen-lockfile so Bun can resolve for Linux in Docker)
COPY package.json bun.lock ./
RUN bun install

# Ensure onnxruntime-node CUDA binaries are installed (Bun might skip post-install scripts)
# Run npm install for onnxruntime-node specifically to trigger CUDA binary download
RUN cd /app && npm install onnxruntime-node@^1.24.1 --no-save || echo "CUDA binaries may already be installed"

# Copy source code and models
COPY src ./src
COPY models ./models
COPY tsconfig.json ./

EXPOSE 3001
CMD ["bun", "run", "src/index.ts"]
