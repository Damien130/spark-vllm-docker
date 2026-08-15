#!/bin/bash
# PROFILE: NVIDIA Nemotron-3-Super-120B-A12B NVFP4
# DESCRIPTION: vLLM serving nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 (NVFP4 quantized LatentMoE)

# NVIDIA-recommended environment variables for NVFP4 on DGX Spark
export VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_NVFP4_GEMM_BACKEND=marlin
export VLLM_TEST_FORCE_FP8_MARLIN=1
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Download reasoning parser into the container
mkdir -p /app
if [ ! -f /app/super_v3_reasoning_parser.py ]; then
    curl -L -o /app/super_v3_reasoning_parser.py https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/resolve/main/super_v3_reasoning_parser.py
fi

vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --moe-backend marlin \
    --mamba_ssm_cache_dtype float16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 262144 \
    --max-num-seqs 10 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --max-cudagraph-capture-size 128 \
    --host 0.0.0.0 \
    --port 8000 \
    --load-format fastsafetensors \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser-plugin /app/super_v3_reasoning_parser.py \
    --reasoning-parser super_v3 \
    --tensor-parallel-size 1 \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"moe_backend\":\"triton\"}" \
    --attention-backend TRITON_ATTN
