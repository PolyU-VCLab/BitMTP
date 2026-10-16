#!/usr/bin/env bash
# Build only; this command neither loads a CUDA context nor executes a kernel.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ARCH=${SM_ARCH:-90}
case "$ARCH" in 86|87|89|90) ;; *) echo "Unsupported SM_ARCH=$ARCH (choose 86, 87, 89, 90)" >&2; exit 2;; esac
NVCC=${NVCC:-nvcc}
# Some distributions expose /usr/bin/nvcc as a symlink; canonicalize it so
# nvcc finds the toolkit include/lib directories relative to its true path.
NVCC=$(realpath "$(command -v "$NVCC")")
KIND=$(basename "$HERE")
case "$KIND" in
  q1g128_kernels) SOURCE=q1g128_kernels_block8.cu; LIB=libbonsai_q1.so ;;
  t2g128_kernels) SOURCE=t2g128_kernels.cu; LIB=libbonsai_t2.so ;;
  *) echo "Unexpected kernel directory: $KIND" >&2; exit 2 ;;
esac
"$NVCC" -O3 -std=c++17 --use_fast_math -Xcompiler -fPIC -shared \
  -arch="sm_${ARCH}" "$HERE/$SOURCE" -o "$HERE/$LIB"
echo "Built $LIB for sm_${ARCH}; GPU numerical validation is separate."
