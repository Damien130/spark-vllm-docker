#!/bin/bash
set -e
# Marlin NVFP4 Fix for DGX Spark (SM121)
# Replaces broken CUTLASS FP4 kernels with Marlin backend
#
# Background: SM121 (DGX Spark GB10) lacks `tcgen05` tensor core instructions
# that datacenter Blackwell (SM100/SM110) has. The default CUTLASS FP4 kernels
# generate unsupported PTX instructions and silently fall back to slower paths.
#
# Marlin dequantizes FP4 to BF16 on the fly using operations that work on SM121.
# Benchmarks show ~16% faster throughput and ~7 GB less memory vs. CUTLASS fallback.
#
# See: https://forums.developer.nvidia.com/t/marlin-fix-nvfp4-actually-works-on-sm121-dgx-spark/365119

echo "=======> Marlin NVFP4 fix applied (SM121/DGX Spark)"
echo "=======> Environment variables set via recipe env section:"
echo "=======>   VLLM_USE_FLASHINFER_MOE_FP4=0"
echo "=======>   VLLM_NVFP4_GEMM_BACKEND=marlin"
echo "=======>   VLLM_TEST_FORCE_FP8_MARLIN=1"
