FROM oven/bun:1 AS base
WORKDIR /app

# Install dependencies
COPY package.json bun.lockb ./
RUN bun install --frozen-lockfile

# Copy source code and models
COPY src ./src
COPY models ./models
COPY tsconfig.json ./

# Expose port
EXPOSE 3001

# Run the application
CMD ["bun", "run", "src/index.ts"]
