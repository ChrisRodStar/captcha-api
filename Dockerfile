FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
WORKDIR /app

# Install Bun
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    npm \
    ca-certificates \
    && curl -fsSL https://bun.sh/install | bash \
    && ln -s /root/.bun/bin/bun /usr/local/bin/bun \
    && rm -rf /var/lib/apt/lists/*

# Note: For GPU support:
# 1. NVIDIA GPU with CUDA 12.x drivers installed on the host (✓ verified working)
# 2. nvidia-container-toolkit installed on the host (✓ verified working)
# 3. runtime: nvidia configured in docker-compose.yml (✓ configured)
# 4. CUDA runtime libraries are installed above (libcublas, libcudnn, etc.)
#    These are required for onnxruntime-node CUDA provider to work

# Install dependencies with Bun (no --frozen-lockfile so Bun can resolve for Linux in Docker)
COPY package.json bun.lock ./
RUN bun install

# Remove the CPU-only onnxruntime-node installed by Bun and reinstall with CUDA support
RUN rm -rf node_modules/onnxruntime-node && \
    npm install onnxruntime-node --onnxruntime-node-install-cuda

# Copy source code and models
COPY src ./src
COPY models ./models
COPY tsconfig.json ./

EXPOSE 3001
CMD ["bun", "run", "src/index.ts"]
