# Reference server environment

The versions below were read from the Linux server used for the final experiments on
September 7, 2026. They document the executed environment rather than a minimum
compatibility claim.

## System

- Operating system: Ubuntu 24.04 family, Linux kernel 7.0.0-28-generic, x86_64
- Python: 3.12.12
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB
- NVIDIA driver: 580.173.02
- CUDA reported by NVIDIA-SMI: 13.0
- CUDA reported by PyTorch: 13.0
- cuDNN reported by PyTorch: 9.15.1

## Core software

Exact Python package versions are listed in `requirements.txt`. The principal runtime
components were PyTorch 2.10.0+cu130, torchvision 0.25.0+cu130, Transformers 5.2.0,
PEFT 0.18.1, bitsandbytes 0.49.2, nnU-Net v2 2.7.0, and Ultralytics 8.4.51.
`openpyxl==3.1.5` is included in the public environment for optional XLSX export; it was
not required by the server-side model training or inference runs.

The final Qwen base model was Qwen3.5-VL 4B stored locally. Base-model files are not
redistributed by this repository and must be obtained under the model provider's terms.
