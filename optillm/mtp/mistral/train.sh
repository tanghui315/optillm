#!/bin/bash

# 基础配置
MODEL_NAME="/home/ubuntu/cjr/models/LLM-Research/Codestral-22B-v0___1"  # 预训练模型路径
OUTPUT_DIR="./sft_outputs"         # 输出目录

# 数据集配置
SFT_DATASET="instruction,multiround"            # SFT数据集名称
SFT_DATASET_DIR="/home/ubuntu/cjr/data/v4"              # SFT数据集目录
PT_DATASET="code_fim"                     # PT数据集名称
PT_DATASET_DIR="/home/ubuntu/cjr/data/codefim-v4"                # PT数据集目录

# 训练参数
BATCH_SIZE=1
GRAD_ACCUMULATION=16
LEARNING_RATE=1e-5
NUM_EPOCHS=3
SAVE_STEPS=500
EVAL_STEPS=100
VAL_SIZE=0.05
# LoRA配置
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LR_RATIO=10.0

# FSDP配置
FSDP="full_shard"
FSDP_CONFIG="fsdp_config.json"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行训练脚本
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ${FSDP_CONFIG} \
    train.py \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --n_future_tokens 2 \
    --template mistral \
    --dataset $SFT_DATASET \
    --dataset_dir $SFT_DATASET_DIR \
    --sft_dataset $SFT_DATASET \
    --sft_dataset_dir $SFT_DATASET_DIR \
    --pt_dataset $PT_DATASET \
    --pt_dataset_dir $PT_DATASET_DIR \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --max_seq_length 4096 \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --val_size $VAL_SIZE \
    --cutoff_len 4096 \
    --preprocessing_num_workers 16 \
    --logging_steps 10 \
    --use_lora True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --loraplus_lr_ratio $LR_RATIO \
    --use_rslora True \
    --bits 4 \
    --bf16 True \
    --do_train \
    --overwrite_output_dir \
    --fsdp ${FSDP} 