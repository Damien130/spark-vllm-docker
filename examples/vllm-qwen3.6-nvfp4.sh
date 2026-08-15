#!/bin/bash
# PROFILE: NVIDIA Qwen3.6-35B-A3B-NVFP4
# DESCRIPTION: vLLM serving nvidia/Qwen3.6-35B-A3B-NVFP4 (NVFP4 quantized MoE)
# NOTE: Requires --apply-mod mods/qwen3.6-nvfp4 for NVIDIA-recommended env vars

# NVIDIA-recommended environment variables for NVFP4 on DGX Spark
export VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_FP8_MOE_BACKEND=flashinfer_cutlass
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUTE_DSL_ARCH=sm_121a

vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --quantization modelopt \
    --trust-remote-code \
    --moe-backend marlin \
    --kv-cache-dtype fp8 \
    --load-format fastsafetensors \
    --attention-backend flashinfer \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --async-scheduling \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 262144 \
    --max-num-batched-tokens 16384 \
    --host 0.0.0.0 \
    --port 8000
