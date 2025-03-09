#!/usr/bin/env python
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
预先使用OpenAI API接口进行样本采样并计算奖励值，生成预处理数据集，以便在GRPO训练中直接使用。
"""

import os
import argparse
import logging
import sys
import torch
import numpy as np
import requests
import time
import json
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, List, Any
from tqdm import tqdm

import datasets
import transformers
from datasets import load_dataset, Dataset
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    set_seed,
)

from open_r1.configs import GRPOConfig
from open_r1.utils import get_tokenizer
from open_r1.rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
)
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template

logger = logging.getLogger(__name__)


@dataclass
class PresamplingArguments:
    """
    预采样脚本的参数。
    """
    model_name_or_path: str = field(
        metadata={"help": "要用于生成的模型的路径或标识符"}
    )
    api_base_url: str = field(
        default="http://localhost:8000/v1",
        metadata={"help": "OpenAI兼容API的基础URL"}
    )
    api_key: Optional[str] = field(
        default="EMPTY",
        metadata={"help": "API密钥（如果需要）"}
    )
    dataset_name: str = field(
        metadata={"help": "用于采样的数据集的名称"}
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "数据集的配置名称"}
    )
    dataset_train_split: str = field(
        default="train",
        metadata={"help": "用于训练的数据集分割"}
    )
    dataset_test_split: str = field(
        default="test",
        metadata={"help": "用于测试的数据集分割"}
    )
    output_dir: str = field(
        metadata={"help": "保存采样数据集的目录"}
    )
    num_generations: int = field(
        default=4,
        metadata={"help": "每个提示生成的样本数量"}
    )
    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "提示的最大长度"}
    )
    max_completion_length: int = field(
        default=1024,
        metadata={"help": "生成完成的最大长度"}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "采样的温度参数"}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "系统提示（如果有）"}
    )
    reward_funcs: List[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "奖励函数列表。可能的值: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    seed: int = field(
        default=42,
        metadata={"help": "随机种子"}
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "错误答案的最小奖励值"}
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "错误答案的最大奖励值"}
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "正确答案的最小奖励值"}
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "正确答案的最大奖励值"}
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "缩放的最大长度"}
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "重复惩罚奖励的n-grams数量"}
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "重复惩罚奖励的最大（负）惩罚"}
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "代码格式奖励的语言",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )
    retry_attempts: int = field(
        default=3,
        metadata={"help": "API调用失败时的重试次数"}
    )
    retry_delay: float = field(
        default=1.0,
        metadata={"help": "重试之间的延迟（秒）"}
    )


def generate_with_openai_api(prompt, api_base_url, api_key, model_name, num_generations, max_tokens, temperature):
    """
    使用OpenAI兼容的API生成回复
    """
    headers = {
        "Content-Type": "application/json",
    }
    
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    
    # 格式化消息
    if isinstance(prompt, list):
        # 对话格式
        messages = prompt
    else:
        # 非对话格式，使用单个用户消息
        messages = [{"role": "user", "content": prompt}]
    
    data = {
        "model": model_name,
        "messages": messages,
        "n": num_generations,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    response = requests.post(
        f"{api_base_url}/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        logger.error(f"API请求失败，状态码: {response.status_code}")
        logger.error(f"响应内容: {response.text}")
        raise Exception(f"API请求失败: {response.text}")
    
    result = response.json()
    completions = [choice["message"]["content"] for choice in result["choices"]]
    
    return completions


def main():
    # 解析命令行参数
    parser = HfArgumentParser((PresamplingArguments, GRPOConfig, ModelConfig))
    presampling_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # 设置随机种子
    set_seed(presampling_args.seed)

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 加载数据集
    logger.info(f"加载数据集: {presampling_args.dataset_name}")
    dataset = load_dataset(presampling_args.dataset_name, name=presampling_args.dataset_config)

    # 加载分词器
    logger.info(f"加载分词器: {model_args.model_name_or_path}")
    tokenizer = get_tokenizer(model_args, training_args)

    # 获取奖励函数
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=presampling_args.cosine_min_value_wrong,
            max_value_wrong=presampling_args.cosine_max_value_wrong,
            min_value_correct=presampling_args.cosine_min_value_correct,
            max_value_correct=presampling_args.cosine_max_value_correct,
            max_len=presampling_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=presampling_args.repetition_n_grams,
            max_penalty=presampling_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=presampling_args.code_language),
        "tag_count": tag_count_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in presampling_args.reward_funcs]

    # 对奖励模型加载它们的分词器
    reward_processing_classes = []
    for func_name in presampling_args.reward_funcs:
        if func_name in ["accuracy", "format", "reasoning_steps", "tag_count", "length", "code", "code_format"]:
            # 这些是自定义函数，不需要分词器
            reward_processing_classes.append(None)
        else:
            # 可能的奖励模型，需要加载对应的分词器
            # 虽然我们这里没有真的使用模型作为奖励函数，但为了保持一致性，我们使用相同的分词器
            tokenizer_for_reward = tokenizer
            if tokenizer_for_reward.pad_token_id is None:
                tokenizer_for_reward.pad_token = tokenizer_for_reward.eos_token
            reward_processing_classes.append(tokenizer_for_reward)

    # 格式化为对话
    def make_conversation(example):
        prompt = []

        if presampling_args.system_prompt is not None:
            prompt.append({"role": "system", "content": presampling_args.system_prompt})

        prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    # 为训练集和测试集创建采样数据
    for split in [presampling_args.dataset_train_split, presampling_args.dataset_test_split]:
        logger.info(f"为 {split} 分割创建采样数据")
        
        split_dataset = dataset[split]
        
        # 用于存储最终的jsonl数据
        final_examples = []
        
        for i, example in enumerate(tqdm(split_dataset, desc=f"采样 {split}")):
            prompt = example["prompt"]
            # 使用OpenAI API生成样本，支持重试
            completions = None
            for attempt in range(presampling_args.retry_attempts):
                try:
                    completions = generate_with_openai_api(
                        prompt=prompt,
                        api_base_url=presampling_args.api_base_url,
                        api_key=presampling_args.api_key,
                        model_name=presampling_args.model_name_or_path,
                        num_generations=presampling_args.num_generations,
                        max_tokens=presampling_args.max_completion_length,
                        temperature=presampling_args.temperature
                    )
                    break
                except Exception as e:
                    if attempt < presampling_args.retry_attempts - 1:
                        logger.warning(f"API调用失败: {e}，尝试重试 ({attempt+1}/{presampling_args.retry_attempts})")
                        time.sleep(presampling_args.retry_delay)
                    else:
                        logger.error(f"所有重试都失败了: {e}")
                        raise
            
            # 格式化生成的文本
            formatted_completions = []
            for completion in completions:
                formatted_completion = completion
                formatted_completions.append(formatted_completion)
            
            # 计算每个生成样本的奖励值
            all_rewards = []
            
            for completion in formatted_completions:
                rewards_for_completion = []
                
                for i, (reward_func, reward_processing_class) in enumerate(
                    zip(reward_funcs, reward_processing_classes)
                ):
                    # 自定义奖励函数
                    reward_kwargs = {key: example[key] for key in example.keys() if key not in ["prompt", "completion"]}
                    reward = reward_func(
                        prompts=[prompt], 
                        completions=[completion], 
                        **reward_kwargs
                    )[0]
                    rewards_for_completion.append(reward)
                
                all_rewards.append(sum(rewards_for_completion))
            
            # 创建completions数组，包含所有生成结果和奖励
            completions_data = []
            for j, (completion, reward) in enumerate(zip(formatted_completions, all_rewards)):
                if is_conversational(example):
                    completion_text = completion[0]["content"]  # 从对话格式中提取文本内容
                else:
                    completion_text = completion
                
                completions_data.append({
                    "text": completion_text,
                    "reward": reward,
                    "index": j
                })
            
            # 创建一条包含原始数据和所有生成结果的记录
            new_example = {}
            
            # 保留原始字段
            for key in example:
                if key != "prompt" and key != "completion":  # 这两个字段将特殊处理
                    new_example[key] = example[key]
            
            # 添加prompt字段（保持原始格式）
            new_example["prompt"] = prompt
            
            # 添加completions数组
            new_example["completions"] = completions_data
            
            final_examples.append(new_example)
        
        # 保存为jsonl格式
        output_path = os.path.join(presampling_args.output_dir, f"sampled_{split}.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in final_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"已保存 {split} 采样数据集到 {output_path}")
        
        # 同时保存为Dataset格式，便于后续处理
        processed_examples = []
        for example in final_examples:
            for completion_data in example["completions"]:
                processed_example = dict(example)  # 复制原始字段
                del processed_example["completions"]  # 删除completions数组
                
                # 添加单个completion和reward
                if is_conversational(example):
                    processed_example["completion"] = [{"role": "assistant", "content": completion_data["text"]}]
                else:
                    processed_example["completion"] = completion_data["text"]
                
                processed_example["reward"] = completion_data["reward"]
                processed_example["generation_idx"] = completion_data["index"]
                
                processed_examples.append(processed_example)
        
        # 创建并保存Dataset格式
        dataset_output_path = os.path.join(presampling_args.output_dir, f"sampled_{split}_dataset")
        os.makedirs(dataset_output_path, exist_ok=True)
        
        dataset_format = Dataset.from_dict({
            key: [example[key] for example in processed_examples if key in example]
            for key in processed_examples[0].keys()
        })
        
        dataset_format.save_to_disk(dataset_output_path)
        logger.info(f"已同时保存Dataset格式数据到 {dataset_output_path}")

if __name__ == "__main__":
    main()
