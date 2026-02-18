# CAPTCHA Solver API

Standalone CAPTCHA solving API using ONNX model for high-performance, concurrent CAPTCHA solving.

## Features

- Fast ONNX-based CAPTCHA solving (~97% accuracy)
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

3. Run the server:

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

Check server health and readiness.

### GET /stats

Get solver statistics.

## Deployment

### Docker Compose (Recommended)

1. Build and start:

```bash
docker-compose up -d
```

2. View logs:

```bash
docker-compose logs -f
```

3. Stop:

```bash
docker-compose down
```

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

## Performance

- Average solve time (CPU): ~50-100ms per CAPTCHA
- Concurrent requests: Limited by CPU resources
- Success rate: ~97%
- Cost: $0 (local processing)
