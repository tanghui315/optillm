#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

# 设置GPU环境
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  # 训练使用前6张卡
#export CUDA_VISIBLE_DEVICES_FOR_VLLM=6,7  # vLLM使用后2张卡
# export VLLM_TENSOR_PARALLEL_SIZE=2  # 必须与VLLM使用的GPU数量匹配

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=6 src/open_r1/grpo.py \
    --config ./grpo_config.yaml
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=6 src/open_r1/faster_grpo.py \
    --config ./config_fast_grpo.yaml


docker run --runtime nvidia --shm-size=32g  --gpus '"device=6,7"' -p 5903:30000 \
    -v /home/ubuntu/33b_coder/models/merged_model:/models \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path /models --host 0.0.0.0 --port 30000 --tp 2 


export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1
 python -m sglang.launch_server --model-path /home/ubuntu/33b_coder/models/merged_model --host 0.0.0.0 --port 30000 --tp 2 --enable-p2p-check 