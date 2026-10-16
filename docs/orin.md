# Jetson Orin

The Orin profiles target Bonsai-1.7B SFT, D4, and D8 on Jetson Orin NX. They use native T2-g128 CUDA kernels, eager PyTorch helpers, and CUDA Graphs with one 4096-token KV segment. They do not use FP8, Triton, xFormers, or `torch.compile`. Prefill uses SDPA; captured decode retains the runtime's explicit attention implementation.

## Environment

Start with JetPack 6.2.1 / L4T 36.4.7, including CUDA 12.6 and cuDNN 9.3, on Jetson Orin NX Super 16GB. The commands below use Python 3.10 and NVIDIA's PyTorch `2.5.0a0+872d972e41.nv24.08` wheel in a separate environment. They do not use the desktop H20 or RTX 3090 requirements.

From the repository root, install the Python environment and cuSPARSELt locally:

```bash
sudo apt-get install -y python3-venv python3-pip libopenblas-dev curl xz-utils
python3 -m venv .venv-orin
source .venv-orin/bin/activate
python -m pip install -r requirements/orin.txt
python -m pip install 'https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl'

curl -fL 'https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.6.3.2-archive.tar.xz' -o /tmp/bitmtp-cusparselt.tar.xz
mkdir -p .venv-orin/cusparselt
tar -xJf /tmp/bitmtp-cusparselt.tar.xz -C .venv-orin/cusparselt --strip-components=1
export LD_LIBRARY_PATH="$PWD/.venv-orin/cusparselt/lib:${LD_LIBRARY_PATH:-}"

python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()'
SM_ARCH=87 NVCC=/usr/local/cuda/bin/nvcc bash src/bonsai_sft/gpu/t2g128_kernels/compile.sh
```

Run the activation and `LD_LIBRARY_PATH` commands again in each new terminal. For offline devices, download the same NVIDIA wheel, cuSPARSELt archive, and ARM64 Python dependencies on a connected host; install the wheels with `--no-index --find-links /path/to/wheels`. The native library must be compiled on the target architecture; an x86-64 `.so` cannot be copied to Jetson. See [NVIDIA's PyTorch guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) for system prerequisites.

## Models

Use the same Hugging Face models and [conversion command](usage.md#convert-weights) as the desktop runtime. Conversion can run on the host before deployment; no separate Orin weights are needed. Transfer the converted `bonsai_gpu_config.json`, `model_state_t2g128.pt`, and the matching model's tokenizer files to the device. Keep the other conversion outputs on the host. These native-only Orin profiles do not load `model_state_fp16.pt`.

| Runtime profile | Model key | Block size | Mode |
| --- | --- | ---: | --- |
| `orin-sft` | `bonsai_1p7_sft` | 8 | `ar` |
| `orin-d4` | `bonsai_1p7_d4` | 4 | `conf`, `spec`, or `hybrid` |
| `orin-d8` | `bonsai_1p7_d8` | 8 | `conf`, `spec`, or `hybrid` |

The SFT block-size field is checkpoint metadata; AR still processes one token per step. Use each profile only with its matching model. The tokenizer directory must come from that same Hugging Face checkpoint, not another model with a similar name.

## Generate

For D8 hybrid generation:

```bash
python -m bonsai_sft.gpu.run_generation \
  --runtime-profile orin-d8 \
  --ckpt /path/to/packed-bonsai-d8 --hf-ckpt /path/to/tokenizer-bonsai-d8 \
  --block-size 8 --mode hybrid \
  --draft-precision native --verify-precision native \
  --threshold 0.9 --min-chain 3 --max-context 4096 \
  --chat --prompt 'Write a short greeting.' --max-new-tokens 128
```

For D4, use the D4 paths, `--runtime-profile orin-d4`, and `--block-size 4`; the hybrid minimum chain remains 3. For SFT AR, use the SFT paths with `--runtime-profile orin-sft --block-size 8 --mode ar`, and omit `--min-chain`.

Add `--check-only` to validate local artifacts and options without initializing CUDA. The full prompt, output budget, and graph workspace must fit the segment. These commands run single-prompt generation; see [decoding modes](usage.md#decode). Runtime profiles leave device power mode and clocks unchanged.
