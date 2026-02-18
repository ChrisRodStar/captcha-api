FROM oven/bun:1-debian
WORKDIR /app

# Install system dependencies for onnxruntime-node (CPU) + healthcheck + npm for CUDA binary installation
RUN apt-get update && apt-get install -y \
    libgomp1 \
    wget \
    gnupg2 \
    curl \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Note: For GPU support:
# 1. NVIDIA GPU with CUDA 12.x drivers installed on the host (✓ verified working)
# 2. nvidia-container-toolkit installed on the host (✓ verified working)
# 3. runtime: nvidia configured in docker-compose.yml (✓ configured)
# 4. CUDA runtime libraries are provided by nvidia-container-toolkit via the host
#    The onnxruntime-node CUDA binaries should work if GPU is accessible

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
