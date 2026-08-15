#!/bin/bash
set -e
cd $WORKSPACE_DIR
curl -L -o super_v3_reasoning_parser.py https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/resolve/main/super_v3_reasoning_parser.py
mkdir -p /app
cp super_v3_reasoning_parser.py /app/super_v3_reasoning_parser.py
