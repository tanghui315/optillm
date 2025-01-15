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
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn
from .mistral_mtp_model import MistralMTPForCausalLM

from ...model.loader import _get_init_kwargs,load_config
from ...model.patcher import patch_config, patch_model
from ...model.model_utils.liger_kernel import apply_liger_kernel
from ...model.model_utils.mod import convert_pretrained_model_to_mod
from ...model.model_utils.unsloth import load_unsloth_pretrained_model
from ...model.adapter import init_adapter
from ...model.model_utils.misc import register_autoclass
from ...extras.misc import count_parameters

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
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)

class MixedDataCollator:
    def __init__(self, tokenizer, model=None, **kwargs):
        self.tokenizer = tokenizer
        self.pt_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        self.sft_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=kwargs.get('label_pad_token_id', -100)
        )
        self.debug_count = 0
        logger.info("MixedDataCollator initialized")

    def __call__(self, features):
        logger.info(f"Processing batch with {len(features)} features")
        
        pt_features = []
        sft_features = []

        for feature in features:
            if feature.get("is_pt", False):
                # 确保数据格式正确
                pt_feature = {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                    "labels": feature["input_ids"].copy() if "labels" not in feature else feature["labels"]
                }
                pt_features.append(pt_feature)
            else:
                sft_features.append(feature)

        logger.info(f"Split into {len(pt_features)} PT and {len(sft_features)} SFT features")

        batch = {}
        if pt_features:
            try:
                pt_batch = self.pt_collator(pt_features)
                # 只在前几个 batch 或出错时打印调试信息
                if self.debug_count < 3:  # 只打印前3个batch
                    print("PT batch keys:", pt_batch.keys())
                    print("PT batch shapes:", {k: v.shape for k, v in pt_batch.items()})
                    self.debug_count += 1
                batch.update(pt_batch)
            except Exception as e:
                print("Error in PT collation:", e)
                print("PT features:", pt_features[0].keys())
                raise e

        if sft_features:
            sft_batch = self.sft_collator(sft_features)
            batch.update(sft_batch)

        return batch


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
        sft_dataset[split] = sft_dataset[split].map(
            lambda x: {**x, "is_pt": False}
        )

    for split in pt_dataset:
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


def load_mtp_model(
    tokenizer,
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
):
    logger.info_rank0("Starting model loading process...")
    init_kwargs = _get_init_kwargs(model_args)
    
    # 添加 FSDP 相关配置
    init_kwargs.update({
        "low_cpu_mem_usage": True,
        "torch_dtype": model_args.compute_dtype,
    })
    
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args)

    setattr(config, "n_future_tokens", 2)
    logger.info_rank0("Set n_future_tokens to 2")

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
        # 添加内存优化相关参数
        load_class = MistralMTPForCausalLM

        if model_args.train_from_scratch:
            # 从头开始训练时使用新架构
            model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
        else:
            model = load_class.from_pretrained(**init_kwargs)

        logger.info_rank0("Model loading completed successfully")

    if not lazy_load:
        # 在应用 LoRA 之前，先解冻 lm_head 和 future_lm_heads
        if is_trainable and finetuning_args.finetuning_type == "lora":
            # 解冻 lm_head
            if hasattr(model, "lm_head"):
                model.lm_head.requires_grad_(True)
            # 解冻 future_lm_heads
            if hasattr(model, "future_lm_heads"):
                for head in model.future_lm_heads:
                    head.requires_grad_(True)

        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)
        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    return model


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
    dataset_module = get_mixed_dataset(template, model_args, data_args, training_args, **tokenizer_module)
    if data_args.template == 'mistral':
        logger.info_rank0("Loading Mistral MTP model")
        model = load_mtp_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    else:
        model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = MixedDataCollator(
        tokenizer=tokenizer,
        model=model if not training_args.predict_with_generate else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
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
