#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=6 src/open_r1/grpo.py \
    --config ./grpo_config.yaml