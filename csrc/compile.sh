#!/usr/bin/env bash
# Build the DCT chunk-major CUDA kernel shared library for Linux
# Requires: CUDA Toolkit 12.x, g++
set -euo pipefail

ARCH="${1:-sm_86}"  # RTX 3060/4060=sm_86, RTX 3090/4090=sm_89, A100=sm_80, H100=sm_90
DIR="$(cd "$(dirname "$0")" && pwd)"
CU="$DIR/dct_chunk_major_kernel.cu"
SO="$DIR/dct_kernel_chunk_major.so"

nvcc -shared -o "$SO" "$CU" -arch="$ARCH" -O2 --use_fast_math
echo "OK -> $SO (arch=$ARCH)"
