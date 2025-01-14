import copy
import math
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Dict, Sequence
import logging
import os
import functools

import torch
import torch.distributed
import transformers
from transformers import Trainer, BitsAndBytesConfig
import datasets
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from llamafactory.data import get_dataset, split_dataset
from transformers import TrainingArguments
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from llamafactory.hparams.model_args import ModelArguments as LModelArguments
from llamafactory.model.loader import load_tokenizer
from mistral_mtp_model import MistralMTPForCausalLM
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
IGNORE_INDEX = -100
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments(LModelArguments):
    trainable: Optional[str] = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    n_future_tokens: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of future tokens to predict in MTP (Multi-Token Prediction). Default is 1 for standard training."
        }
    )
    max_seq_length: Optional[int] = field(
        default=8192,
        metadata={
            "help": "The maximum total sequence length for input sequences. Sequences longer than this will be truncated."
        }
    )
    lora_rank: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[int] = field(default=32)
    padding_side: Optional[str] = field(default="right")
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )

    loraplus_lr_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A)."},
    )
    loraplus_lr_embedding: float = field(
        default=1e-5,
        metadata={"help": "LoRA plus learning rate for lora embedding layers."},
    )

    use_dora: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use the weight-decomposed lora method (DoRA)."},
    )

    modules_to_save: Optional[str] = field(
        default=None)  # "embed_tokens,lm_head"
    use_lora: Optional[bool] = field(default=False)
    model_name_or_path: Optional[str] = field(default="mistral")
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    model_max_length: Optional[int] = field(
        default=None,
        init=False,
        metadata={
            "help": "The maximum input length for model, derived from `cutoff_len`. Do not specify it."},
    )
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={
            "help": "Which scaling strategy should be adopted for the RoPE embeddings."},
    )


@dataclass
class TrainingArguments(TrainingArguments):
    # cache_dir: Optional[str] = field(default=None)
    predict_with_generate: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    sft_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    sft_dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    pt_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    pt_dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether to plot the training loss curve."}
    )


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)


def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(
            os.path.join(checkpoint_dir, 'completed'))
        if is_completed:
            return None  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
                max_step = max(max_step, int(
                    filename.replace(PREFIX_CHECKPOINT_DIR + '-', '')))
        if max_step == 0:
            return None
        latest_ckpt_dir = os.path.join(
            checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return latest_ckpt_dir
    return None  # first training


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu()
                          for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


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
            label_pad_token_id=kwargs.get('label_pad_token_id', -100),
            padding=True
        )

    def __call__(self, features):
        torch.cuda.empty_cache()
        
        pt_features = []
        sft_features = []

        for feature in features:
            if feature.get("is_pt", False):
                # 移除SFT特有的字段
                pt_feature = {k: v for k, v in feature.items()
                              if k in ["input_ids", "attention_mask", "labels"]}
                pt_features.append(pt_feature)
            else:
                sft_features.append(feature)

        batch = {}
        if pt_features:
            pt_batch = self.pt_collator(pt_features)
            # pt_batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
            #            for k, v in pt_batch.items()}
            batch.update(pt_batch)

        if sft_features:
            sft_batch = self.sft_collator(sft_features)
            # sft_batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
            #             for k, v in sft_batch.items()}
            batch.update(sft_batch)
        # 确保batch中包含所有必要的字段
        # if "attention_mask" not in batch:
        #     batch["attention_mask"] = batch["input_ids"].ne(
        #         self.tokenizer.pad_token_id)

        return batch


def create_loraplus_optimizer(model, model_args, training_args):
    default_lr = training_args.learning_rate
    loraplus_lr = training_args.learning_rate * model_args.loraplus_lr_ratio
    embedding_lr = model_args.loraplus_lr_embedding
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_param_names = [
        name for name in decay_parameters if "bias" not in name]
    param_dict: Dict[str, List["torch.nn.Parameter"]] = {
        "lora_a": [],
        "lora_b": [],
        "lora_b_nodecay": [],
        "embedding": [],
    }
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora_embedding_B" in name:
                param_dict["embedding"].append(param)
            elif "lora_B" in name or param.ndim == 1:
                if name in decay_param_names:
                    param_dict["lora_b"].append(param)
                else:
                    param_dict["lora_b_nodecay"].append(param)
            else:
                param_dict["lora_a"].append(param)

    optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(
        training_args)
    param_groups = [
        dict(params=param_dict["lora_a"], lr=default_lr,
             weight_decay=training_args.weight_decay),
        dict(params=param_dict["lora_b"], lr=loraplus_lr,
             weight_decay=training_args.weight_decay),
        dict(params=param_dict["lora_b_nodecay"],
             lr=loraplus_lr, weight_decay=0.0),
        dict(params=param_dict["embedding"], lr=embedding_lr,
             weight_decay=training_args.weight_decay),
    ]
    optimizer = optim_class(param_groups, **optim_kwargs)
    print(
        "Using LoRA+ optimizer with loraplus lr ratio {:.2f}.".format(model_args.loraplus_lr_ratio))
    return optimizer


def build_model(model_args, training_args, checkpoint_dir):
    def print_detailed_memory():
        if training_args.local_rank == 0:
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i} Memory Details:")
                print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f}GB")
                print(f"  Reserved:  {torch.cuda.memory_reserved(i) / 1024**3:.2f}GB")
                print(f"  Max Allocated: {torch.cuda.max_memory_allocated(i) / 1024**3:.2f}GB")
                print(f"  Max Reserved:  {torch.cuda.max_memory_reserved(i) / 1024**3:.2f}GB")
                # 清除最大值统计
                torch.cuda.reset_peak_memory_stats(i)

    if not model_args.use_lora:
        assert model_args.bits in [16, 32]
    compute_dtype = (torch.bfloat16 if training_args.bf16 else torch.float16)
    print_detailed_memory()
    print("\n=== Before loading quantization config ===")

    quantization_config = None
    if model_args.use_lora and (model_args.bits == 4 or model_args.bits == 8):
        if model_args.bits == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif model_args.bits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quant,
                bnb_4bit_quant_type=model_args.quant_type,
                bnb_4bit_quant_storage=compute_dtype,
            )

    print_detailed_memory()
    print("\n=== Before loading config ===")

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
    )

    print_detailed_memory()
    print("\n=== Before loading model ===")

    # 设置模型的最大序列长度
    if model_args.max_seq_length:
        config.max_length = model_args.max_seq_length
        if training_args.local_rank == 0:
            print(f"Setting model max_length to {model_args.max_seq_length}")

    target_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    config.torch_dtype = target_dtype
    if training_args.local_rank == 0:
        print(f"Setting model dtype to {target_dtype}")

    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = model_args.attn_implementation
        if training_args.local_rank == 0:
            print(f"Using attention implementation: {config._attn_implementation}")

    config.use_cache = False
    config.n_future_tokens = model_args.n_future_tokens
    if model_args.n_future_tokens > 1 and training_args.local_rank == 0:
        print(f"Using Multi-Token Prediction with {model_args.n_future_tokens} future tokens")
    
    configure_rope(config, model_args, True)

    # 使用MistralMTPForCausalLM
    model = MistralMTPForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        quantization_config=quantization_config,
        torch_dtype=target_dtype,
        # device_map="auto",  # 让模型自动处理设备分配
        offload_folder="offload",  # 设置模型权重卸载目录
        offload_state_dict=True,  # 启用状态字典卸载
        # torch_dtype="auto",  # 自动选择数据类型
    )

    print_detailed_memory()
    print("\n=== After loading model ===")

    optimizer = None
    if model_args.loraplus_lr_ratio is not None:
        optimizer = create_loraplus_optimizer(model, model_args, training_args)

    if compute_dtype == torch.float16 and model_args.bits == 4:
        if torch.cuda.is_bf16_supported():
            logger.info('='*80)
            logger.info(
                'Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            logger.info('='*80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.torch_dtype = target_dtype

    # Prepare for kbit training if needed
    if model_args.use_lora:
        print("\n=== Before preparing for kbit training ===")
        print_detailed_memory()

        if model_args.bits < 16:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing)

        print("\n=== After preparing for kbit training ===")
        print_detailed_memory()

        if checkpoint_dir is not None:
            model = PeftModel.from_pretrained(
                model, checkpoint_dir, is_trainable=True)

        print("\n=== After loading LoRA ===")
        print_detailed_memory()

        logger.info(f'Init LoRA modules...')
        target_modules = model_args.trainable.split(',')
        modules_to_save = model_args.modules_to_save
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(',')
        lora_rank = model_args.lora_rank
        lora_dropout = model_args.lora_dropout
        lora_alpha = model_args.lora_alpha
        if training_args.local_rank == 0:
            print(f'lora_rank: {lora_rank}')
            print(f'lora_alpha: {lora_alpha}')
            print(f'lora_dropout: {lora_dropout}')
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_rslora=model_args.use_rslora,
            use_dora=model_args.use_dora,
            modules_to_save=modules_to_save)
        model = get_peft_model(model, peft_config)

    
    print(f'target_dtype: {target_dtype}')
    # for name, module in model.named_modules():
    #     if isinstance(module, LoraLayer):
    #         if training_args.bf16:
    #             module = module.to(target_dtype)
    #     if 'lm_head' in name or 'embed_tokens' in name:
    #         if hasattr(module, 'weight'):
    #             if module.weight.dtype == torch.float32:
    #                 module = module.to(target_dtype)

    # # 确保所有参数使用相同的数据类型
    # model = model.to(target_dtype)

    return model, optimizer


def configure_rope(config, model_args: "ModelArguments", is_trainable: bool) -> None:
    if model_args.rope_scaling is None:
        return

    if not hasattr(config, "rope_scaling"):
        print("Current model does not support RoPE scaling.")
        return

    if model_args.model_max_length is not None:
        if is_trainable and model_args.rope_scaling == "dynamic":
            print(
                "Dynamic NTK scaling may not work well with fine-tuning. "
                "See: https://github.com/huggingface/transformers/pull/24653"
            )

        current_max_length = getattr(config, "max_position_embeddings", None)
        if current_max_length and model_args.model_max_length > current_max_length:
            print(f"Enlarge max model length from {current_max_length} to {model_args.model_max_length}.")
            setattr(config, "max_position_embeddings",
                    model_args.model_max_length)
            scaling_factor = float(
                math.ceil(model_args.model_max_length / current_max_length))
        else:
            print(
                "Input length is smaller than max length. Consider increase input length.")
            scaling_factor = 1.0
    else:
        scaling_factor = 2.0

    setattr(config, "rope_scaling", {
            "type": model_args.rope_scaling, "factor": scaling_factor})
    print(
        f"Using {model_args.rope_scaling} scaling strategy and setting scaling factor to {scaling_factor}"
    )


def get_mixed_dataset(template, model_args, data_args, training_args, **kwargs):
    """获取并合并SFT和PT数据集"""
    def parse_dataset_names(dataset_str):
        """解析数据集名称，将逗号分隔的字符串转换为列表"""
        if not dataset_str:
            return []
        return [name.strip() for name in dataset_str.split(",")]
    
    # 启用流式加载
    # data_args.streaming = True
    
    # 创建SFT数据集的配置
    sft_args = copy.deepcopy(data_args)
    sft_args.dataset = parse_dataset_names(training_args.sft_dataset)
    sft_args.dataset_dir = training_args.sft_dataset_dir

    # 创建PT数据集的配置
    pt_args = copy.deepcopy(data_args)
    pt_args.dataset = parse_dataset_names(training_args.pt_dataset)
    pt_args.dataset_dir = training_args.pt_dataset_dir

    # 分别获取数据集
    sft_dataset = get_dataset(template, model_args,
                             sft_args, training_args, stage="sft", **kwargs)
    pt_dataset = get_dataset(template, model_args,
                            pt_args, training_args, stage="pt", **kwargs)

    # 为数据集添加标识
    for split in sft_dataset:
        sft_dataset[split] = sft_dataset[split].map(
            lambda x: {**x, "is_pt": False},
            remove_columns=sft_dataset[split].column_names
        )

    for split in pt_dataset:
        pt_dataset[split] = pt_dataset[split].map(
            lambda x: {**x, "is_pt": True},
            remove_columns=pt_dataset[split].column_names
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


class MTPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 获取原始模型
        unwrapped_model = self.accelerator.unwrap_model(model)
        
        # 如果不是MTP模式，直接使用父类的compute_loss
        if not hasattr(unwrapped_model, "n_future_tokens") or unwrapped_model.n_future_tokens == 1:
            return super().compute_loss(model, inputs, return_outputs)
            
        # 直接使用model.forward的输出
        outputs = model(**inputs)
        loss = outputs.loss

        # 如果需要label smoothing或其他特殊处理
        if self.label_smoother is not None:
            loss = self.label_smoother(outputs, inputs["labels"])

        return (loss, outputs) if return_outputs else loss


def train():
    # 添加内存监控
    def print_gpu_memory():
        if training_args.local_rank == 0:  # 只在主进程打印
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated, "
                  f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB reserved")
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if training_args.local_rank == 0:
        print_gpu_memory()  # 监控初始内存
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    if training_args.local_rank == 0:
        logger.info('='*100)
        logger.info(training_args)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    logger.info("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    logger.info("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    logger.info("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    if training_args.local_rank == 0:
        logger.info("Load tokenizer from {} over.".format(
            model_args.model_name_or_path))

    resume_from_checkpoint_dir = get_last_checkpoint(training_args.output_dir)

    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    # 使用新的get_mixed_dataset替代原来的get_dataset
    data_modules = get_mixed_dataset(
        template, model_args, data_args, training_args, **tokenizer_module)
    # data_modules = split_dataset(dataset, data_args, training_args)

    model, optimizer = build_model(
        model_args, training_args, resume_from_checkpoint_dir)
    # 使用MixedDataCollator
    data_collator = MixedDataCollator(
        tokenizer=tokenizer,
        model=model if not training_args.predict_with_generate else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    print("data_info:")
    print(data_modules)
    trainer = MTPTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        ddp_find_unused_parameters=False,
        **data_modules
    )
    if optimizer is not None:
        trainer.optimizer = optimizer
    print(f'use_lora: {model_args.use_lora}')
    print(f'resume_from_checkpoint_dir: {resume_from_checkpoint_dir}')
    if model_args.use_lora:
        trainer.add_callback(SavePeftModelCallback)
    print("start train")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint_dir)
    trainer.save_state()
    if not model_args.use_lora:
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir)

    print_gpu_memory()  # 监控模型加载后的内存

if __name__ == "__main__":
    train()