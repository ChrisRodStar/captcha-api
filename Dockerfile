FROM oven/bun:1-debian AS base
WORKDIR /app

# Install system dependencies for onnxruntime-node
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

# Copy source code and models
COPY src ./src
COPY models ./models
COPY tsconfig.json ./

# Expose port
EXPOSE 3001

# Run the application
CMD ["bun", "run", "src/index.ts"]
