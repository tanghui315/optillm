import os
from typing import Optional, Union, Tuple, List
import torch
import torch.nn as nn
from transformers.models.mistral.modeling_mistral import MistralModel, MistralForCausalLM, MistralDecoderLayer, MistralRMSNorm
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
from ...extras.logging import get_logger
from ...model.loader import _get_init_kwargs,load_config
from ...model.patcher import patch_config, patch_model
from ...model.model_utils.liger_kernel import apply_liger_kernel
from ...model.model_utils.unsloth import load_unsloth_pretrained_model
from ...model.adapter import init_adapter
from ...model.model_utils.misc import register_autoclass
from ...extras.misc import count_parameters
import math
from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

logger = get_logger(__name__)

class LightweightMTPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),  # 降维
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, config.hidden_size)    # 升维
        )
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 与原始层参数共享
        if hasattr(config, 'share_attention_weights') and config.share_attention_weights:
            self.attention = None  # 实际使用时会共享主模型的注意力层

    def forward(self, hidden_states, base_attention_output):
        """
        hidden_states: 主模型的隐藏状态 [batch, seq_len, hidden_size]
        base_attention_output: 主模型最后一层的注意力输出
        """
        if self.attention is not None:
            # 如果有独立注意力层
            x = self.attention(hidden_states)
        else:
            # 直接复用主模型的注意力输出
            x = base_attention_output
            
        projected = self.projection(x)
        return self.norm(x + projected)  # 残差连接

class MistralMTPModel(MistralModel):
    def __init__(self, config):
        super().__init__(config)
        self.n_future_tokens = getattr(config, "n_future_tokens", 1)
        
        if self.n_future_tokens > 1:
            # 轻量级预测头（重命名为 mtp_modules 以保持命名一致）
            self.mtp_modules = nn.ModuleList([
                LightweightMTPHead(config)
                for _ in range(self.n_future_tokens)
            ])

    def forward(self, input_ids, **kwargs):
        # 获取基础模型输出
        base_outputs = super().forward(input_ids, **kwargs)
        last_hidden_state = base_outputs.last_hidden_state
        
        if hasattr(self, 'mtp_modules'):
            all_outputs = [last_hidden_state]
            for head in self.mtp_modules:
                # 使用最后一层的注意力输出作为基础
                head_output = head(
                    last_hidden_state, 
                    base_outputs.attentions[-1] if base_outputs.attentions else last_hidden_state
                )
                all_outputs.append(head_output)
            
            # 拼接所有输出 [batch, seq_len, n_future_tokens+1, hidden_size]
            stacked = torch.stack(all_outputs, dim=2)
            return BaseModelOutputWithPast(
                last_hidden_state=stacked,
                hidden_states=base_outputs.hidden_states,
                attentions=base_outputs.attentions
            )
        return base_outputs

# 用于语言模型的MTP模型
class MistralMTPForCausalLM(MistralForCausalLM):
    _no_split_modules = ["MistralDecoderLayer"]
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    # _tied_weights_keys = ["lm_head.weight", "future_lm_heads.weight"]
    # _tp_plan = {"lm_head": "colwise_rep", "future_lm_heads": "colwise_rep"}
    # # 添加这个类属性来帮助 PEFT 识别模型结构
    # base_model_prefix = "model"
    
    def __init__(self, config):
        super().__init__(config)
        # 添加调试信息
        logger.info_rank0("Initializing MistralMTPForCausalLM")
        # 替换原始的MistralModel为MistralMTPModel
        self.model = MistralMTPModel(config)
        logger.info_rank0(f"Created MistralMTPModel with n_future_tokens: {self.model.n_future_tokens}")
        logger.info_rank0(f"Has mtp_modules: {hasattr(self.model, 'mtp_modules')}")
    
    # 添加这个方法来帮助 PEFT 找到正确的层
    def get_decoder(self):
        """Helper method to help PEFT locate the decoder"""
        return self.model

    def get_wrapped_policy(self):
        """返回 FSDP 包装策略"""
        from torch.distributed.fsdp.wrap import (
            transformer_auto_wrap_policy,
            _module_wrap_policy,
        )
        from functools import partial
        from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

        # 定义需要单独包装的 MTP 模块类型
        mtp_module_types = {
            nn.ModuleDict,  # MTP 模块的容器类型
            MistralDecoderLayer,  # MTP 模块中的解码层
            nn.Linear,  # MTP 模块中的线性层
        }

        # 定义基础模型的包装策略
        base_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={MistralDecoderLayer},
            no_wrap_modules={
                MistralMTPModel,
                nn.ModuleList,
                nn.Linear
            }
        )

        # 自定义包装策略
        def custom_wrap_policy(module, recurse, **kwargs):
            # 优先处理 MTP 模块
            if any(isinstance(module, t) for t in mtp_module_types):
                return True
            
            # 其他模块使用基础策略
            return base_policy(module, recurse, **kwargs)

        return custom_wrap_policy

    def _set_gradient_checkpointing(self, module, value=False):
        """确保 MTP 模块也启用梯度检查点"""
        if isinstance(module, MistralMTPModel):
            module.gradient_checkpointing = value
            # 同时设置 MTP 模块中的梯度检查点
            for mtp_module in module.mtp_modules:
                mtp_module.transformer.gradient_checkpointing = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # 如果不是MTP模式，使用父类的forward
        if not hasattr(self.model, "n_future_tokens") or self.model.n_future_tokens == 1:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 直接使用 self.model 的输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]  # 这里已经包含了所有需要的信息
        
        # 检查hidden_states是否有NaN
        if torch.isnan(hidden_states).any():
            raise ValueError("NaN detected in hidden_states before decoder layer")
        
        hidden_states = hidden_states.permute(0, 2, 1, 3)  # [batch, seq_len, n_future_tokens+1, hidden_size]
        
        # 检查permute后的hidden_states
        if torch.isnan(hidden_states).any():
            print(f"\n[Warning] Found NaN in hidden_states after permute")
            print(f"- hidden_states shape after permute: {hidden_states.shape}")
            print(f"- hidden_states stats after permute:")
            print(f"  * min={hidden_states.min().item() if not torch.isnan(hidden_states.min()) else 'nan'}")
            print(f"  * max={hidden_states.max().item() if not torch.isnan(hidden_states.max()) else 'nan'}")
            print(f"  * mean={hidden_states.mean().item() if not torch.isnan(hidden_states.mean()) else 'nan'}")
            print(f"  * % of NaN: {torch.isnan(hidden_states).float().mean().item() * 100:.2f}%")

        # 计算 logits
        logits = []
        for i in range(self.model.n_future_tokens + 1):  # +1 是因为包括原始模型的输出
            step_hidden = hidden_states[:, :, i]  # [batch, seq_len, hidden_size]
            step_logits = self.lm_head(step_hidden)  # [batch, seq_len, vocab_size]
            logits.append(step_logits.unsqueeze(1))
        #   print(f"Step {i} logits shape: {step_logits.shape}")
        logits = torch.cat(logits, dim=1)  # [batch, n_future_tokens+1, seq_len, vocab_size]
        #print(f"Final logits shape: {logits.shape}")

        loss = None
        if labels is not None:
            logits = logits.to(dtype=self.config.torch_dtype)
            loss_fct = CrossEntropyLoss(ignore_index=-100)  # 在这里指定ignore_index
            losses = []
            
            # 计算所有模块的 loss（包括原始模型）
            for i in range(self.model.n_future_tokens + 1):
                # 所有预测头都预测t+1位置，但基于不同输入
                pred_logits = logits[:, i, :-(i+1)]  # 预测长度递减
                targets = labels[:, (i+1):]          # 目标始终是下一个token
                # 统一掩码：前i+1个位置无效
                mask = torch.zeros_like(targets, dtype=torch.bool)
                mask[:, :i] = True 
                mask |= (targets == -100)
                targets = targets.masked_fill(mask, -100)
                
                # 在损失计算前添加有效性检查
                valid_targets = (targets != -100).sum()
                if valid_targets == 0:
                    print(f"跳过预测头 {i}，无有效目标")
                    continue  # 跳过当前预测头的损失计算
                
                reshaped_logits = pred_logits.reshape(-1, pred_logits.size(-1))
                reshaped_targets = targets.reshape(-1)
                
                # 检查logits是否有异常值
                print(f"Reshaped logits stats:")
                print(f"- min={reshaped_logits.min().item():.4f}, max={reshaped_logits.max().item():.4f}")
                print(f"- mean={reshaped_logits.mean().item():.4f}, std={reshaped_logits.std().item():.4f}")
                print(f"- has_nan: {torch.isnan(reshaped_logits).any().item()}")
                print(f"- has_inf: {torch.isinf(reshaped_logits).any().item()}")
                
                # 使用梯度裁剪来防止梯度爆炸
                pred_logits = torch.clamp(pred_logits, min=-100, max=100)
                
                step_loss = loss_fct(
                    reshaped_logits,
                    reshaped_targets
                )
                print(f"Step {i} loss: {step_loss.item()}")
                losses.append(step_loss)
            
            if len(losses) > 0:
                # 使用指数衰减的权重
                weights = torch.exp(-torch.arange(len(losses), device=losses[0].device) * 0.5)
                weights = weights / weights.sum()
                loss = torch.stack(losses) @ weights
                print(f"Final weighted loss: {loss.item()}")

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    
    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)
    
    if is_trainable:
        # 通过 base_model 访问原始模型
        base_model = model.base_model if hasattr(model, "base_model") else model
        if hasattr(base_model.model, "mtp_modules"):
            for i, module in enumerate(base_model.model.mtp_modules):
                for param in module.parameters():
                    param.requires_grad_(True)
                logger.info_rank0(f"MTP Module {i} parameters set to trainable")
    
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
    # 输出 model_args.compute_dtype
    logger.info_rank0(f"model_args.compute_dtype: {model_args.compute_dtype}")
    
    return model