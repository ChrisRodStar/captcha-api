# GPU Setup Guide

## Why GPU is Failing

Based on the error logs, GPU is failing for these reasons:

1. **No GPU device access**: Docker container can't see GPU devices
   - Error: `Failed to open file: "/sys/class/drm/card0/device/vendor"`
   - Solution: Configure Docker to pass GPU devices to container

2. **Missing CUDA runtime**: CUDA libraries aren't available in container
   - Error: `Failed to load shared library libonnxruntime_providers_shared.so`
   - Solution: Use nvidia-docker2 to provide CUDA from host

3. **Docker not configured for GPU**: docker-compose.yml doesn't enable GPU passthrough
   - Solution: Add GPU configuration to docker-compose.yml

## Requirements

To use GPU in Docker, you need:

1. **NVIDIA GPU** with CUDA 12.x compatible drivers installed on the host
2. **nvidia-docker2** installed on the host system
3. **Docker Compose** configured to pass GPU to container

## Setup Steps

### 1. Check if you have an NVIDIA GPU

```bash
nvidia-smi
```

If this command works and shows your GPU, you have NVIDIA drivers installed.

### 2. Install nvidia-docker2

Follow the official guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

Quick install (Ubuntu/Debian):
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 3. Enable GPU in docker-compose.yml

Uncomment the GPU configuration in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Or use the runtime method:
```yaml
runtime: nvidia
```

### 4. Rebuild and Start

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
docker compose logs captcha-api
```

### 5. Verify GPU is Working

Check the logs - you should see:
```
✅ GPU acceleration active using: CUDA
```

Or check the API:
```bash
curl http://localhost:3001/gpu
```

## Alternative: CPU-Only Mode

If you don't have a GPU or don't want to set it up, the system will automatically fall back to CPU. Set in `.env`:

```
USE_GPU=false
```

## Troubleshooting

### Error: "nvidia-docker2 not found"
- Install nvidia-docker2 (see step 2 above)

### Error: "No GPU devices found"
- Verify GPU is visible: `nvidia-smi`
- Check Docker can see GPU: `docker run --rm --gpus all nvidia/cuda:12.0.0-base nvidia-smi`
- Ensure docker-compose.yml has GPU configuration enabled

### Error: "CUDA version mismatch"
- Ensure host has CUDA 12.x installed
- onnxruntime-node requires CUDA 12.x (CUDA 11 is no longer supported)

### Still using CPU despite GPU setup
- Check logs for GPU initialization errors
- Verify `USE_GPU` is not set to `false` in `.env`
- Check `/gpu` endpoint to see actual status
