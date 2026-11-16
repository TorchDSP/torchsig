# TorchSig: Docker

Run TorchSig using Docker. We have two Dockerfiles, one CPU only and the other GPU capable.

## Base TorchSig Image (`Dockerfile`)
- CPU only
- Image size: ~7 GB
```bash
docker build -t torchsig -f docker/Dockerfile .
docker run -it torchsig
```

## GPU TorchSig Image (`Dockerfile.gpu`)
- Ubuntu 24.04
- NVIDIA CUDA 12.6.3
- Python 3.12
- Image size: ~7 GB
```bash
docker build -t torchsig-gpu -f docker/Dockerfile.gpu .
docker run -it torchsig-gpu
```

## Tips
- To run with GPU support add `--gpus all` in `docker run` command
- The build is split into two layers: dependencies are installed first and
  cached on `pyproject.toml`, so editing source only re-runs a fast editable
  reinstall instead of re-downloading torch. Rebuilds are quickest when the
  Docker layer cache is available (`--cache-from` in CI).
- OpenCV is installed as `opencv-python-headless` (no display/GL deps).
- The GPU image pulls PyTorch CUDA 12.6 wheels; the CPU image pulls the
  CPU-only wheels.