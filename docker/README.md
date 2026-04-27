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
- Ubuntu 22.04
- NVIDIA CUDA 11.8.0
- Python 3.10
- Image size: ~7 GB
```bash
docker build -t torchsig-gpu -f docker/Dockerfile.gpu .
docker run -it torchsig-gpu
```

## Tips
- To run with GPU support add `--gpus all` in `docker run` command