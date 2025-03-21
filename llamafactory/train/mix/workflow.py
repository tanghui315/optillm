# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

import copy
import os
from typing import TYPE_CHECKING, List, Optional, Dict, Any

import torch
import torch.nn as nn
from .mistral_mtp_model import MistralMTPForCausalLM

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps, get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import MixedTrainer
from transformers import AutoModelForCausalLM,DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
import datasets
import time
from dataclasses import dataclass
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)

@dataclass
class MixedDataCollator(DataCollatorForSeq2Seq):
    def __init__(
        self, 
        model=None,
        pad_to_multiple_of=None,  # for shift short attention
        label_pad_token_id=None,
        block_diag_attn=False,
        attn_implementation=None,
        compute_dtype=None,
        tokenizer=None
    ):
        super().__init__(
            model=model,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
            tokenizer=tokenizer,
            padding=True,
        )
        self.pt_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 过滤掉不需要的字段
        filtered_features = []
        
        # 先检查是否为 PT 数据
        is_pt = features and features[0].get("is_pt", False)
        
        # 根据数据类型选择需要保留的字段
        if is_pt:
            # PT 数据只需要 input_ids 和 attention_mask
            needed_keys = {'input_ids', 'attention_mask'}
        else:
            # SFT 数据需要 input_ids, attention_mask 和 labels
            needed_keys = {'input_ids', 'attention_mask', 'labels'}
        
        for feature in features:
            filtered_feature = {k: v for k, v in feature.items() if k in needed_keys}
            filtered_features.append(filtered_feature)

        # 根据之前保存的 is_pt 标记来决定使用哪个 collator
        if is_pt:
            # if filtered_features and len(filtered_features) > 0:
            #     logger.info_rank0(f"PT Sample feature keys: {list(filtered_features[0].keys())}")
            # PT数据只需要input_ids，DataCollatorForLanguageModeling会自动处理labels
            return self.pt_collator.torch_call(filtered_features)
        else:
            # if filtered_features and len(filtered_features) > 0:
            #     logger.info_rank0(f"SFT Sample feature keys: {list(filtered_features[0].keys())}")
            # SFT数据使用父类处理
            return super().__call__(filtered_features)


def get_mixed_dataset(template, model_args, data_args, training_args, **kwargs):
    """获取并合并SFT和PT数据集"""
    def parse_dataset_names(dataset_str):
        """解析数据集名称，将逗号分隔的字符串转换为列表"""
        if not dataset_str:
            return []
        return [name.strip() for name in dataset_str.split(",")]
    
    # 创建SFT数据集的配置
    sft_args = copy.deepcopy(data_args)
    sft_args.dataset = parse_dataset_names(data_args.sft_dataset)
    sft_args.dataset_dir = data_args.sft_dataset_dir

    # 创建PT数据集的配置
    pt_args = copy.deepcopy(data_args)
    pt_args.dataset = parse_dataset_names(data_args.pt_dataset)
    pt_args.dataset_dir = data_args.pt_dataset_dir

    # 分别获取数据集
    sft_dataset = get_dataset(template, model_args,
                             sft_args, training_args, stage="sft", **kwargs)
    pt_dataset = get_dataset(template, model_args,
                            pt_args, training_args, stage="pt", **kwargs)

    # 为数据集添加标识
    for split in sft_dataset:
        logger.info_rank0(f"SFT dataset '{split}' columns before mapping: {sft_dataset[split].column_names}")   
        # 只删除 images 和 videos 列
        # columns_to_remove = ['images', 'videos']
        sft_dataset[split] = sft_dataset[split].map(
            lambda x: {**x, "is_pt": False},
            # remove_columns=columns_to_remove
        )
        # 添加验证代码
        logger.info_rank0(f"SFT dataset '{split}' columns after mapping: {sft_dataset[split].column_names}")

    for split in pt_dataset:
        logger.info_rank0(f"PT dataset '{split}' columns before mapping: {pt_dataset[split].column_names}")
        pt_dataset[split] = pt_dataset[split].map(
            lambda x: {**x, "is_pt": True}
        )

    # 合并数据集
    combined_dataset = {}
    datasets_to_concat = []
    if "train_dataset" in sft_dataset:
        datasets_to_concat.append(sft_dataset["train_dataset"])
    if "train_dataset" in pt_dataset:
        datasets_to_concat.append(pt_dataset["train_dataset"])
    
    if not datasets_to_concat:
        raise ValueError("No training datasets available")
    
    train_dataset = datasets.concatenate_datasets(datasets_to_concat)
    logger.info(f"Combined training dataset size: {len(train_dataset)}")
    
    # 分割训练集和验证集
    if data_args.val_size > 0:
        val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
        split_dataset = train_dataset.train_test_split(
            test_size=val_size, 
            seed=training_args.seed
        )
        combined_dataset["train_dataset"] = split_dataset["train"]
        combined_dataset["eval_dataset"] = split_dataset["test"]
    else:
        combined_dataset["train_dataset"] = train_dataset

    return combined_dataset


def run_mixed(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    # dataset_module = get_mixed_dataset(template, model_args, data_args, training_args, **tokenizer_module)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    # if data_args.template == 'mistral':
    #     logger.info_rank0("Loading Mistral MTP model")
    #     model = load_mtp_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    # else:
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = MixedDataCollator(
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        tokenizer=tokenizer,
    )
    logger.info(f"Created data collator: {data_collator}")

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Initialize our Trainer
    trainer = MixedTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if trainer.is_world_process_zero() and finetuning_args.plot_loss:
        plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
