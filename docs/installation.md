# Installation

Deployment guides cover NVIDIA H20, and RTX 3090 and Jetson Orin NX for Bonsai-1.7B SFT, D4, and D8. Follow the platform-specific dependencies. For H20, start with a Python 3.10 environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/h20.txt
python -m pip install --no-deps -e .
```

`requirements/h20.txt` pairs PyTorch 2.10.0+cu128 with xFormers 0.0.34. A PyTorch wheel does not include the complete CUDA development toolkit. Install a compatible toolkit separately and make its `nvcc` available; the H20 kernel build was checked with CUDA toolkit 12.4.

```bash
SM_ARCH=90 NVCC=/usr/local/cuda/bin/nvcc bash src/bonsai_sft/gpu/t2g128_kernels/compile.sh
```

The script accepts `NVCC` for an explicit compiler path. No models or dependencies are downloaded by this build. No compiled libraries are stored in Git. Use an editable installation so the runtime can load the newly built library from its package directory.

RTX 3090 uses `requirements/rtx3090.txt` and `SM_ARCH=86`, with BF16 drafting. Follow the [RTX 3090 guide](rtx3090.md); do not substitute the H20 dependency set.

For Jetson Orin NX, follow [the Orin guide](orin.md) for the NVIDIA PyTorch wheel, separate Python environment, and SM87 build. Do not install the desktop H20 or RTX 3090 requirements on Jetson.

For training, use the H20 development environment and additionally install:

```bash
python -m pip install -e '.[training]'
```

Check `nvidia-smi` and `torch.cuda.is_available()` before generation.
