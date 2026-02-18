FROM oven/bun:1-debian
WORKDIR /app

# Install system dependencies for onnxruntime-node (CPU) + healthcheck
RUN apt-get update && apt-get install -y \
    libgomp1 \
    wget \
    gnupg2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies with Bun (no --frozen-lockfile so Bun can resolve for Linux in Docker)
COPY package.json bun.lock ./
RUN bun install

# Copy source code and models
COPY src ./src
COPY models ./models
COPY tsconfig.json ./

EXPOSE 3001
CMD ["bun", "run", "src/index.ts"]
