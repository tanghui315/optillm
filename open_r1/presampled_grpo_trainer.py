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
使用预采样数据的GRPO训练器
"""

import os
import textwrap
import warnings
import logging
from collections import defaultdict
from typing import Any, Callable, Optional, Union, List, Dict

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather_object
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl import GRPOTrainer
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

logger = logging.getLogger(__name__)

# 定义奖励函数类型
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class PreSampledGRPOTrainer(GRPOTrainer):
    """
    使用预采样数据的GRPO训练器，不需要vllm。
    
    该训练器假设数据集中已包含采样的完成结果和计算好的奖励值。
    
    参数与GRPOTrainer相同，但添加了一些特定于预采样的参数。
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]] = None,  # 可以为None，因为奖励已预计算
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # 调用父类初始化
        super().__init__(
            model=model,
            # 如果没有提供reward_funcs，创建一个空列表（因为我们使用预计算的奖励）
            reward_funcs=reward_funcs if reward_funcs is not None else [],
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        
        # 强制设置use_vllm为False，因为我们不使用vllm
        self.use_vllm = False
        
        # 验证数据集格式是否正确
        self._verify_dataset_format(train_dataset)
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                for split_dataset in eval_dataset.values():
                    self._verify_dataset_format(split_dataset)
            else:
                self._verify_dataset_format(eval_dataset)
    
    def _verify_dataset_format(self, dataset):
        """验证数据集是否包含所需的列"""
        if dataset is None:
            return
        
        required_columns = ["prompt", "completion", "reward"]
        if isinstance(dataset, Dataset):
            sample = dataset[0]
        else:  # IterableDataset
            for sample in dataset:
                break
            
        missing_columns = [col for col in required_columns if col not in sample]
        if missing_columns:
            raise ValueError(f"数据集缺少必要的列: {missing_columns}. 请确保数据集包含这些列。")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算GRPO损失，使用预采样的数据。
        
        与原始GRPOTrainer不同，此版本不会进行采样，而是直接使用数据集中的采样结果。
        """
        if return_outputs:
            raise ValueError("PreSampledGRPOTrainer不支持返回输出")

        device = self.accelerator.device
        
        # 获取输入数据
        prompts = [x["prompt"] for x in inputs]
        completions = [x["completion"] for x in inputs]
        rewards = torch.tensor([x["reward"] for x in inputs], dtype=torch.float32, device=device)
        
        # 将prompt文本格式化
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        
        # 对prompts进行编码
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        
        # 对completions进行编码
        completions_text = []
        for completion in completions:
            if is_conversational(completion):
                # 如果是对话格式，我们只需要assistant的回复内容
                completions_text.append(completion[0]["content"])
            else:
                # 非对话格式，直接使用completion
                completions_text.append(completion)
        
        completion_inputs = self.processing_class(
            completions_text, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
        )
        completion_inputs = super()._prepare_inputs(completion_inputs)
        
        # 获取completion的token id
        completion_ids = completion_inputs["input_ids"]
        
        # 拼接prompt和completion
        prompt_inputs_repeated = prompt_inputs["input_ids"]
        prompt_completion_ids = torch.cat([prompt_inputs_repeated, completion_ids], dim=1)
        
        # 获取prompt长度
        prompt_length = prompt_inputs["input_ids"].size(1)
        
        # 用于计算每个token的对数概率
        def get_per_token_logps(model, input_ids, num_logits_to_keep):
            # 我们给`num_logits_to_keep`加1，因为后面会排除最后一个logit
            logits = model(input_ids, num_logits_to_keep=num_logits_to_keep + 1).logits  # (B, L, V)
            logits = logits[:, :-1, :]  # (B, L-1, V)，排除最后一个logit，它对应于下一个token的预测
            
            # 计算输入token的对数概率
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids[:, -num_logits_to_keep:]):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)
        
        # 计算数据的completion部分中需要保留的logits数量
        num_logits_to_keep = completion_ids.size(1)
        
        # 计算模型的每个token的对数概率
        per_token_logps = get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep)
        
        # 使用推理模式计算参考模型的每个token的对数概率
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids, num_logits_to_keep)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep)
        
        # 计算KL散度
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        # 处理EOS（结束符）token的mask
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # 计算每组样本的平均奖励和标准差
        # 但因为我们使用预采样数据，每个样本已经有自己的奖励，这里不需要分组
        # 我们仍然可以计算规范化，但直接使用样本的奖励值
        mean_grouped_rewards = rewards.mean()
        std_grouped_rewards = rewards.std()
        
        # 归一化奖励以计算优势
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        # x - x.detach()允许保留x的梯度
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # 记录指标
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        
        # 记录奖励
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.item())
        
        # 记录KL散度
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        
        return loss
