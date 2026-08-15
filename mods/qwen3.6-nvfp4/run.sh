#!/bin/bash
set -e
# Mod for Qwen3.6-35B-A3B-NVFP4
# Applies NVIDIA-recommended environment variables for NVFP4 quantization
# on DGX Spark (Blackwell architecture).

echo "=======> Qwen3.6-NVFP4 mod applied"
echo "=======> Environment variables set via recipe env section:"
echo "=======>   VLLM_USE_FLASHINFER_MOE_FP4=0"
echo "=======>   VLLM_FP8_MOE_BACKEND=flashinfer_cutlass"
echo "=======>   FLASHINFER_DISABLE_VERSION_CHECK=1"
echo "=======>   CUTE_DSL_ARCH=sm_121a"
echo "=======>   HF_HOME=/hf_cache"
echo "=======> Use --quantization modelopt --moe-backend marlin"
