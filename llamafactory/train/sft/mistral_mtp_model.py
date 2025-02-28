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


class MTPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 拼接后的投影层
        self.projection = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
        # 简化版的transformer decoder层
        self.decoder_layer = MistralDecoderLayer(config, layer_idx=0)  # layer_idx设为0或其他值
        
    def forward(
        self, 
        input_tokens: torch.Tensor,  # 当前token的隐藏状态，作为查询
        hidden_states: torch.Tensor,  # 上一个MTP模块传来的隐藏状态，作为键值记忆
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_tokens: [batch, seq_len, hidden_size] 当前要处理的token
            hidden_states: [batch, seq_len, hidden_size] 上一个MTP模块的状态
            attention_mask: 注意力掩码
            position_ids: 位置编码
        Returns:
            updated_hidden: 更新后的隐藏状态，传给下一个MTP模块
        """
        # 规范化输入
        normed_hidden = self.hidden_norm(hidden_states)
        normed_input = self.input_norm(input_tokens)
        
        # 拼接并投影
        concat_features = torch.cat([normed_hidden, normed_input], dim=-1)
        projected = self.projection(concat_features)
        
        # 通过transformer decoder层，按照官方实现传递参数
        layer_outputs = self.decoder_layer(
            hidden_states=projected,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,  # 训练时不使用缓存
            output_attentions=False,
            use_cache=False,
            cache_position=None
        )
        
        # MistralDecoderLayer返回一个元组，第一个元素是更新后的hidden_states
        updated_hidden = layer_outputs[0]
        
        return updated_hidden

class MistralMTPModel(MistralModel):
    def __init__(self, config):
        super().__init__(config)
        self.n_future_tokens = getattr(config, "n_future_tokens", 1)
        
        if self.n_future_tokens > 1:
            # 创建多个MTP模块
            self.mtp_blocks = nn.ModuleList([
                MTPBlock(config)
                for _ in range(self.n_future_tokens - 1)
            ])

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        # 获取基础模型输出
        base_outputs = super().forward(input_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs)
        last_hidden_state = base_outputs.last_hidden_state
        
        if hasattr(self, 'mtp_blocks') and self.training:
            all_hidden_states = [last_hidden_state]
            curr_hidden = last_hidden_state
            
            # 获取批处理中每个序列的有效长度（基于注意力掩码）
            if attention_mask is not None:
                # 计算每个序列的有效长度
                valid_seq_lengths = attention_mask.sum(dim=1).to(torch.int)
            else:
                # 如果没有提供注意力掩码，假设所有序列长度都等于输入的长度
                valid_seq_lengths = torch.full(
                    (input_ids.shape[0],), 
                    input_ids.shape[1], 
                    dtype=torch.int, 
                    device=input_ids.device
                )
            
            # 顺序处理每个MTP模块
            for i, mtp_block in enumerate(self.mtp_blocks):
                batch_size = input_ids.shape[0]
                valid_batch_indices = []  # 记录哪些批次项有足够长度处理
                
                # 为有效批次准备数据
                valid_inputs = []
                valid_hiddens = []
                valid_masks = []
                valid_positions = []
                
                for b in range(batch_size):
                    # 当前序列需要至少有i+2个token（当前+下一个）才能处理
                    if valid_seq_lengths[b] > i + 2:
                        valid_batch_indices.append(b)
                
                if not valid_batch_indices:
                    # 如果没有有效批次，则跳过此MTP模块
                    continue
                    
                # 构建有效批次的张量
                for b in valid_batch_indices:
                    # 计算此批次项的有效长度
                    valid_len = valid_seq_lengths[b].item() - (i + 1)
                    
                    # 准备输入数据
                    if input_ids is not None:
                        # 取当前+未来的token作为输入
                        inp = self.embed_tokens(input_ids[b, i+1:valid_seq_lengths[b]]).unsqueeze(0)
                        valid_inputs.append(inp)
                    
                    # 准备隐藏状态
                    # 注意：需要确保长度匹配，取min防止越界
                    valid_len = min(valid_len, curr_hidden.shape[1] - i - 1)
                    h = curr_hidden[b, :(valid_len)].unsqueeze(0)
                    valid_hiddens.append(h)
                    
                    # 准备注意力掩码和位置ID
                    if attention_mask is not None:
                        m = attention_mask[b, i+1:i+1+valid_len].unsqueeze(0)
                        valid_masks.append(m)
                    
                    if position_ids is not None:
                        p = position_ids[b, i+1:i+1+valid_len].unsqueeze(0)
                        valid_positions.append(p)
                
                # 如果有有效数据，处理此MTP模块
                if valid_inputs and valid_hiddens:
                    # 将列表合并为批处理张量
                    batch_inputs = torch.cat(valid_inputs, dim=0)
                    batch_hiddens = torch.cat(valid_hiddens, dim=0)
                    
                    batch_masks = None
                    if valid_masks:
                        batch_masks = torch.cat(valid_masks, dim=0)
                        
                    batch_positions = None
                    if valid_positions:
                        batch_positions = torch.cat(valid_positions, dim=0)
                    
                    # 确保所有张量长度一致
                    min_len = min(batch_inputs.shape[1], batch_hiddens.shape[1])
                    batch_inputs = batch_inputs[:, :min_len]
                    batch_hiddens = batch_hiddens[:, :min_len]
                    
                    if batch_masks is not None:
                        batch_masks = batch_masks[:, :min_len]
                    if batch_positions is not None:
                        batch_positions = batch_positions[:, :min_len]
                    
                    # 通过MTP块处理
                    mtp_output = mtp_block(
                        batch_inputs,
                        batch_hiddens,
                        attention_mask=batch_masks,
                        position_ids=batch_positions
                    )
                    
                    # 创建一个新的空隐藏状态，大小与前一个相同
                    new_hidden = torch.zeros_like(curr_hidden)
                    
                    # 只更新有效批次的有效部分
                    for idx, b in enumerate(valid_batch_indices):
                        valid_len = min(mtp_output.shape[1], new_hidden.shape[1])
                        new_hidden[b, :valid_len] = mtp_output[idx, :valid_len]
                    
                    all_hidden_states.append(new_hidden)
                    curr_hidden = new_hidden
                else:
                    # 如果没有有效数据，复制前一个隐藏状态
                    all_hidden_states.append(curr_hidden)
            
            # 将所有输出堆叠 [batch, n_future_tokens, seq_len, hidden_size]
            stacked = torch.stack(all_hidden_states, dim=1)
            
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
        logger.info_rank0(f"Has mtp_blocks: {hasattr(self.model, 'mtp_blocks')}")
    
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
            for mtp_block in module.mtp_blocks:
                mtp_block.transformer.gradient_checkpointing = value

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
        # 如果不是MTP模式或不在训练中，使用父类的forward
        if not hasattr(self.model, "n_future_tokens") or self.model.n_future_tokens == 1 or not self.training:
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
        
        # 获取模型输出
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
        
        hidden_states = outputs[0]  # [batch, n_future_tokens, seq_len, hidden_size]
        
        # 计算所有时间步的logits
        logits = []
        for i in range(self.model.n_future_tokens):
            step_hidden = hidden_states[:, i]  # [batch, seq_len, hidden_size]
            step_logits = self.lm_head(step_hidden)  # [batch, seq_len, vocab_size]
            logits.append(step_logits)
        
        main_logits = logits[0]  # 主模型的logits
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            
            # 确保标签和logits维度匹配 - 主损失
            seq_len = min(main_logits.size(1), labels.size(1))
            main_loss = loss_fct(main_logits[:, :seq_len].reshape(-1, main_logits.size(-1)), 
                                 labels[:, :seq_len].reshape(-1))
            
            # 计算MTP损失
            mtp_loss = 0.0
            mtp_count = 0  # 计算有效MTP模块数量
            
            for i in range(1, self.model.n_future_tokens):
                # 对于每个MTP模块，预测的是未来的token
                pred_logits = logits[i][:, :-i] if i < logits[i].size(1) else logits[i]  # 安全裁剪
                
                if i < labels.size(1):  # 确保标签有足够长度
                    target_labels = labels[:, i:]  # 目标是第i个未来token
                    
                    # 计算两者的共同长度
                    common_len = min(pred_logits.size(1), target_labels.size(1))
                    
                    if common_len > 0:  # 确保有共同区域可以计算损失
                        # 裁剪到相同长度
                        pred_logits_trimmed = pred_logits[:, :common_len]
                        target_labels_trimmed = target_labels[:, :common_len]
                        
                        # 检查是否有有效标签（非-100）
                        valid_labels = (target_labels_trimmed != -100).sum() > 0
                        
                        if valid_labels:
                            # 计算当前MTP模块的损失
                            mtp_step_loss = loss_fct(
                                pred_logits_trimmed.reshape(-1, pred_logits_trimmed.size(-1)),
                                target_labels_trimmed.reshape(-1)
                            )
                            mtp_loss += mtp_step_loss
                            mtp_count += 1
            
            # 合并损失，使用权重因子
            mtp_weight = 0.3  # 可配置的权重
            loss = main_loss
            
            # 只有当有有效MTP损失时才添加
            if mtp_count > 0:
                loss = loss + mtp_weight * mtp_loss / mtp_count
            
            # 记录损失值用于调试（可选）
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"检测到异常损失值: {loss}，主损失: {main_loss}，MTP损失: {mtp_loss}")
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=main_logits,  # 只返回主模型的logits用于生成
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
        if hasattr(base_model.model, "mtp_blocks"):
            for i, mtp_block in enumerate(base_model.model.mtp_blocks):
                for param in mtp_block.parameters():
                    param.requires_grad_(True)
                logger.info_rank0(f"MTP Block {i} parameters set to trainable")
    
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