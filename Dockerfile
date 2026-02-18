FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
WORKDIR /app

# Install Node.js 20.x (required for modern packages)
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Note: For GPU support:
# 1. NVIDIA GPU with CUDA 12.x drivers installed on the host (✓ verified working)
# 2. nvidia-container-toolkit installed on the host (✓ verified working)
# 3. runtime: nvidia configured in docker-compose.yml (✓ configured)
# 4. CUDA runtime libraries are installed above (libcublas, libcudnn, etc.)
#    These are required for onnxruntime-node CUDA provider to work

# Install dependencies with npm (CUDA binaries are downloaded automatically)
COPY package.json ./
RUN npm install && npm install -g tsx

# Copy source code and models
COPY src ./src
COPY models ./models
COPY tsconfig.json ./

EXPOSE 3001
CMD ["node", "--loader", "tsx", "src/index.ts"]
