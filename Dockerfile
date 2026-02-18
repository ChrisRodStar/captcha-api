FROM oven/bun:1-debian
WORKDIR /app

# Install system dependencies for onnxruntime-node (CPU) + healthcheck
RUN apt-get update && apt-get install -y \
    libgomp1 \
    wget \
    gnupg2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Note: For GPU support, you need:
# 1. NVIDIA GPU with CUDA 12.x drivers installed on the host
# 2. nvidia-docker2 installed on the host: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
# 3. Uncomment the GPU runtime configuration in docker-compose.yml
# 4. The CUDA libraries will be provided by the host via nvidia-docker2
#    No need to install CUDA in the container - it uses the host's CUDA installation

# Install dependencies with Bun (no --frozen-lockfile so Bun can resolve for Linux in Docker)
COPY package.json bun.lock ./
RUN bun install

# Copy source code and models
COPY src ./src
COPY models ./models
COPY tsconfig.json ./

EXPOSE 3001
CMD ["bun", "run", "src/index.ts"]
