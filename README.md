# CAPTCHA Solver API

Standalone CAPTCHA solving API using ONNX model for high-performance, concurrent CAPTCHA solving.

## Features

- Fast ONNX-based CAPTCHA solving (~97% accuracy)
- GPU acceleration support (CUDA/DirectML) with automatic CPU fallback
- Concurrent request handling
- Batch processing endpoint
- Health checks and statistics
- Zero cost per CAPTCHA

## Setup

1. Install dependencies:

```bash
bun install
```

2. Ensure model files are in `./models/`:
   - `captcha_model.onnx`
   - `captcha_model_metadata.json`

3. GPU Configuration:
   - GPU acceleration is **enabled by default**
   - Set `USE_GPU=false` in `.env` to disable GPU and use CPU only
   - **For NVIDIA GPUs (Linux/Windows):** Install CUDA 12.x and cuDNN
   - **For Windows:** DirectML is automatically available (works with AMD/NVIDIA/Intel GPUs)
   - The system will automatically fallback to CPU if GPU is not available

4. Run the server:

```bash
bun run src/index.ts
```

## API Endpoints

### POST /solve

Solve a single CAPTCHA.

**Request:**

```json
{
  "image": "base64_encoded_image",
  "id": "optional_request_id"
}
```

**Response:**

```json
{
  "success": true,
  "code": "A3B7",
  "confidence": 0.987,
  "method": "ONNX"
}
```

### POST /solve/batch

Solve multiple CAPTCHAs concurrently.

**Request:**

```json
{
  "captchas": [
    { "image": "base64_1", "id": "user1" },
    { "image": "base64_2", "id": "user2" }
  ]
}
```

**Response:**

```json
{
  "results": [
    { "id": "user1", "success": true, "code": "A3B7", "confidence": 0.987 },
    { "id": "user2", "success": true, "code": "K9M2", "confidence": 0.992 }
  ]
}
```

### GET /health

Check server health and readiness. Includes GPU status information.

**Response:**
```json
{
  "status": "ok",
  "ready": true,
  "gpu": {
    "enabled": true,
    "available": true,
    "provider": "cuda",
    "availableProviders": ["cuda", "cpu"],
    "requestedProviders": ["cuda", "cpu"]
  },
  "stats": {
    "totalAttempts": 100,
    "successfulDecodes": 97,
    "failures": 3,
    "averageConfidence": 0.985
  }
}
```

### GET /stats

Get solver statistics.

### GET /gpu

Get detailed GPU status information.

**Response:**
```json
{
  "enabled": true,
  "available": true,
  "provider": "cuda",
  "availableProviders": ["cuda", "cpu"],
  "requestedProviders": ["cuda", "cpu"]
}
```

**GPU Status Fields:**
- `enabled`: Whether GPU was requested via `USE_GPU` environment variable
- `available`: Whether a GPU execution provider is actually being used
- `provider`: The execution provider currently in use (`cuda`, `dml`, `cpu`, etc.)
- `availableProviders`: All execution providers available on the system
- `requestedProviders`: Execution providers that were requested (in priority order)

## Deployment

### Docker Compose (Recommended)

1. Build and start:

```bash
docker-compose up -d
```

2. **If you've updated the code and need to rebuild** (to avoid using cached images):

```bash
# Stop containers
docker-compose down

# Rebuild without cache
docker-compose build --no-cache

# Start containers
docker-compose up -d
```

3. View logs:

```bash
docker-compose logs -f captcha-api
```

4. Stop:

```bash
docker-compose down
```

**Note:** The `USE_GPU` environment variable from your `.env` file will be passed to the container. Make sure to rebuild if you change this setting.

### Cloudflare Tunnel

1. Install cloudflared on your Proxmox server
2. Run: `cloudflared tunnel --url http://localhost:3001`
3. Or create a permanent tunnel:

```bash
cloudflared tunnel create captcha-api
cloudflared tunnel route dns captcha-api captcha.yourdomain.com
cloudflared tunnel run captcha-api
```

### Local Development

```bash
bun run dev
```

## GPU Setup

### NVIDIA GPU (CUDA)

**Linux:**
1. Install CUDA 12.x toolkit and cuDNN
2. Verify installation: `nvidia-smi`
3. Set `USE_GPU=true` in `.env`

**Windows:**
1. Install CUDA 12.x toolkit and cuDNN from NVIDIA
2. Verify installation: `nvidia-smi` in PowerShell
3. Set `USE_GPU=true` in `.env`

### Windows DirectML (Alternative)

DirectML works automatically on Windows 10/11 with compatible GPUs (AMD/NVIDIA/Intel). No additional drivers needed beyond standard Windows GPU drivers. Set `USE_GPU=true` in `.env`.

### Verification

Check the server logs on startup to see which execution provider is being used:
- `Available execution providers: cuda, cpu` (CUDA available)
- `Available execution providers: dml, cpu` (DirectML available)
- `Available execution providers: cpu` (CPU only)

## Performance

- Average solve time (CPU): ~50-100ms per CAPTCHA
- Average solve time (GPU): ~10-30ms per CAPTCHA (varies by GPU)
- Concurrent requests: Limited by CPU/GPU resources
- Success rate: ~97%
- Cost: $0 (local processing)
