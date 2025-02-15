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
class MistralMTPModel(MistralModel):
    
    def __init__(self, config):
        if not hasattr(config, 'n_future_tokens'):
            config.n_future_tokens = 1  # 设置默认值
        # # 使用config中的torch_dtype，如果没有则默认使用bfloat16
        # config.torch_dtype = getattr(config, "torch_dtype", torch.bfloat16)
        super().__init__(config)
        self.n_future_tokens = getattr(config, "n_future_tokens", 1)  # 默认为1，兼容原始模型
        
        if self.n_future_tokens > 1:
            # 获取原模型最后一层的引用
            last_layer = self.layers[-1] if len(self.layers) > 0 else None
            
            self.mtp_modules = nn.ModuleList([
                nn.ModuleDict({
                    'projection': nn.Linear(config.hidden_size * 2, config.hidden_size),
                    'pro_norm': MistralRMSNorm(config.hidden_size, eps=1e-5),
                    'transformer': self._create_mtp_layer(config, layer_idx=len(self.layers)+i, src_layer=last_layer)
                }) for i in range(self.n_future_tokens)
            ])
            # 调用 _init_mtp_module 对每个 MTP 模块进行初始化，确保与原模型一致
            for module in self.mtp_modules:
                self._init_mtp_module(module, config)
            
    def _init_mtp_module(self, module, config):
        # 对projection层使用更小的初始化范围
        nn.init.normal_(module['projection'].weight, mean=0.0, std=config.initializer_range * 0.1)

    def _create_mtp_layer(self, config, layer_idx: int, src_layer: Optional[nn.Module] = None):
        new_layer = MistralDecoderLayer(config, layer_idx)
        
        # 如果源层存在，则复制权重
        if src_layer is not None:
            # 深拷贝所有可学习参数
            new_layer.load_state_dict(src_layer.state_dict(), strict=False)
            
            # 特殊处理位置相关的参数（如旋转位置编码）
            if hasattr(new_layer.self_attn, 'rotary_emb'):
                with torch.no_grad():
                    new_layer.self_attn.rotary_emb.inv_freq.copy_(
                        src_layer.self_attn.rotary_emb.inv_freq
                    )
                
        return new_layer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # 添加NaN检测函数
        def check_nan(tensor: torch.Tensor, name: str, module_idx: Optional[int] = None) -> bool:
            if torch.isnan(tensor).any():
                msg = f"\n[Error] Found NaN in {name}"
                if module_idx is not None:
                    msg += f" for MTP module {module_idx}"
                msg += f"\nShape: {tensor.shape}"
                msg += f"\nStats: min={tensor.min().item() if not torch.isnan(tensor.min()) else 'nan'}"
                msg += f", max={tensor.max().item() if not torch.isnan(tensor.max()) else 'nan'}"
                msg += f", mean={tensor.mean().item() if not torch.isnan(tensor.mean()) else 'nan'}"
                msg += f"\nNaN percentage: {torch.isnan(tensor).float().mean().item() * 100:.2f}%"
                logger.error(msg)
                raise ValueError(msg)
            return False

        # 使用autocast来确保正确的数据类型
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        with torch.autocast(device_type=device.type, dtype=self.config.torch_dtype):
            if self.n_future_tokens == 1:
                return super().forward(
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
                    **flash_attn_kwargs,
                )
            
            # MTP模式的forward逻辑
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if self.gradient_checkpointing and self.training and use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
                use_cache = False

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            if use_cache:
                if past_key_values is None:
                    past_key_values = DynamicCache()
            else:
                past_key_values = None  # 确保 use_cache=False 时彻底禁用缓存
            
            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )

            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )

            hidden_states = inputs_embeds

            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
                
            # 通过主干网络
            # decoder layers
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            for decoder_layer in self.layers[: self.config.num_hidden_layers]:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
            # 添加调试信息
            print(f"[MistralMTPModel] Initial use_cache: {use_cache}")
            print(f"[MistralMTPModel] Initial past_key_values: {type(past_key_values) if past_key_values is not None else None}")

            for i, decoder_layer in enumerate(self.layers):
                print(f"\n[Layer {i}/{len(self.layers)-1}] Before decoder_layer:")
                print(f"  - hidden_states shape: {hidden_states.shape}")
                print(f"  - attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
                print(f"  - position_ids shape: {position_ids.shape if position_ids is not None else None}")
                print(f"  - use_cache: {use_cache}")
                
                # 检查 causal_mask
                if isinstance(attention_mask, torch.Tensor):
                    print(f"  - causal_mask shape: {attention_mask.shape}")
                    print(f"  - causal_mask unique values: {torch.unique(attention_mask).tolist()}")
                
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values if (use_cache and isinstance(past_key_values, (Cache, DynamicCache))) else None,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                hidden_states = self.norm(hidden_states)
                
                # 检查trunk_output
                check_nan(hidden_states, "trunk_output")
                
                # 添加调试信息
                print(f"[Layer {i}/{len(self.layers)-1}] After decoder_layer:")
                print(f"  - output shape: {layer_outputs[0].shape}")
                if past_key_values is not None:
                    try:
                        if isinstance(past_key_values, (Cache, DynamicCache)):
                            key_cache = past_key_values.key_cache
                            if isinstance(key_cache, dict) and i in key_cache:
                                key = past_key_values.key_cache[i]
                                value = past_key_values.value_cache[i]
                                print(f"  - updated key shape: {key.shape if key is not None else None}")
                                print(f"  - updated value shape: {value.shape if value is not None else None}")
                    except Exception as e:
                        print(f"Warning: Error checking cache for MTP module {i}: {str(e)}")
            
            # 通过预测头
            latents = []
            prev_output = None  # 用于存储上一个 MTP Module 的输出
            # 获取当前token和目标token的表示
            current_hidden = hidden_states # 当前token的隐层表示
            for i, mtp_module in enumerate(self.mtp_modules):        
                # RMSNorm处理
                if prev_output is not None:
                    # 直接使用前一个模块的输出，形成预测链
                    current_hidden = prev_output
                
                # 检查current_hidden
                check_nan(current_hidden, "current_hidden", i)
                
                norm_hidden = self.norm(current_hidden)
                norm_target = self.norm(inputs_embeds)
                
                # 检查norm输出
                check_nan(norm_hidden, "norm_hidden", i)
                check_nan(norm_target, "norm_target", i)
                
                # Concatenate并投影
                concat_features = torch.cat([norm_hidden, norm_target], dim=-1)
                check_nan(concat_features, "concat_features", i)
                
                projected = mtp_module['projection'](concat_features)
                projected = torch.clamp(projected, min=-1e4, max=1e4)  # 限制数值范围
                projected = mtp_module['pro_norm'](projected)
                check_nan(projected, "projected", i)
                
                # 为 projected 生成新的 position_embeddings
                curr_position_embeddings = self.rotary_emb(projected, position_ids)
                
                # 检查position_embeddings
                if isinstance(curr_position_embeddings, tuple):
                    cos, sin = curr_position_embeddings
                    check_nan(cos, "position_embeddings_cos", i)
                    check_nan(sin, "position_embeddings_sin", i)
                else:
                    check_nan(curr_position_embeddings, "position_embeddings", i)
                
                # 检查attention_mask和causal_mask
                if causal_mask is not None and torch.isnan(causal_mask).any():
                    print(f"\n[Warning] Found NaN in causal_mask for MTP module {i}")
                    print(f"- causal_mask shape: {causal_mask.shape}")
                    print(f"- causal_mask stats:")
                    print(f"  * min={causal_mask.min().item() if not torch.isnan(causal_mask.min()) else 'nan'}")
                    print(f"  * max={causal_mask.max().item() if not torch.isnan(causal_mask.max()) else 'nan'}")
                    print(f"  * mean={causal_mask.mean().item() if not torch.isnan(causal_mask.mean()) else 'nan'}")
                    print(f"  * % of NaN: {torch.isnan(causal_mask).float().mean().item() * 100:.2f}%")
                elif causal_mask is None:
                    print(f"\n[Warning] causal_mask is None for MTP module {i}")
                
                # 通过Transformer Block
                # 检查transformer的输入
                print(f"\n[Debug] Transformer inputs for MTP module {i}:")
                print(f"- projected stats:")
                print(f"  * shape: {projected.shape}")
                print(f"  * has_nan: {torch.isnan(projected).any().item()}")
                print(f"  * min={projected.min().item() if not torch.isnan(projected.min()) else 'nan'}")
                print(f"  * max={projected.max().item() if not torch.isnan(projected.max()) else 'nan'}")
                print(f"  * mean={projected.mean().item() if not torch.isnan(projected.mean()) else 'nan'}")
                
                print(f"- position_ids stats:")
                if position_ids is not None:
                    print(f"  * shape: {position_ids.shape}")
                    print(f"  * has_nan: {torch.isnan(position_ids).any().item()}")
                    print(f"  * values: {position_ids[0, :10].tolist()}")  # 只显示前10个位置
                else:
                    print(f"  * position_ids is None")
                
                print(f"- cache_position stats:")
                if cache_position is not None:
                    print(f"  * shape: {cache_position.shape}")
                    print(f"  * has_nan: {torch.isnan(cache_position).any().item()}")
                    print(f"  * values: {cache_position[:10].tolist()}")  # 只显示前10个位置
                else:
                    print(f"  * cache_position is None")
                
                print(f"- attention_mask stats:")
                if attention_mask is not None:
                    print(f"  * shape: {attention_mask.shape}")
                    print(f"  * has_nan: {torch.isnan(attention_mask).any().item()}")
                    print(f"  * values sample: {attention_mask[0, :10].tolist()}")  # 只显示第一个样本的前10个位置
                else:
                    print(f"  * attention_mask is None")
                
                print(f"- curr_position_embeddings stats:")
                if isinstance(curr_position_embeddings, tuple):
                    cos, sin = curr_position_embeddings
                    print(f"  * cos shape: {cos.shape}")
                    print(f"  * cos has_nan: {torch.isnan(cos).any().item()}")
                    print(f"  * sin shape: {sin.shape}")
                    print(f"  * sin has_nan: {torch.isnan(sin).any().item()}")
                else:
                    print(f"  * shape: {curr_position_embeddings.shape}")
                    print(f"  * has_nan: {torch.isnan(curr_position_embeddings).any().item()}")
                
                # 添加调试信息
                print(f"\n[MTP Module {i}] Before transformer:")
                print(f"  - projected shape: {projected.shape}")
                print(f"  - use_cache: {use_cache}")

                # 检查缓存状态
                layer_idx = len(self.layers) + i  # 计算MTP模块的层索引
                if past_key_values is not None:
                    try:
                        if isinstance(past_key_values, (Cache, DynamicCache)):
                            key_cache = past_key_values.key_cache
                            if isinstance(key_cache, dict) and layer_idx in key_cache:
                                key = key_cache[layer_idx]
                                value = past_key_values.value_cache[layer_idx]
                                print(f"  - key shape: {key.shape if key is not None else None}")
                                print(f"  - value shape: {value.shape if value is not None else None}")
                    except Exception as e:
                        print(f"Warning: Error checking cache for MTP module {i}: {str(e)}")
                
                transformer_output = mtp_module['transformer'](
                    projected,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values if (use_cache and isinstance(past_key_values, (Cache, DynamicCache))) else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=curr_position_embeddings,
                    **flash_attn_kwargs,
                )
                
                # 检查transformer_output
                check_nan(transformer_output[0], "transformer_output", i)
                
                # 添加调试信息
                print(f"[MTP Module {i}] After transformer:")
                print(f"  - output shape: {transformer_output[0].shape}")
                if past_key_values is not None:
                    layer_idx = len(self.layers) + i
                    if past_key_values is not None:
                        try:
                            if isinstance(past_key_values, (Cache, DynamicCache)):
                                key_cache = past_key_values.key_cache
                                if isinstance(key_cache, dict) and layer_idx in key_cache:
                                    key = past_key_values.key_cache[layer_idx]
                                    value = past_key_values.value_cache[layer_idx]
                                    print(f"  - updated key shape: {key.shape if key is not None else None}")
                                    print(f"  - updated value shape: {value.shape if value is not None else None}")
                        except Exception as e:
                            print(f"Warning: Error checking cache for MTP module {i}: {str(e)}")
                
                prev_output = transformer_output[0]
                output = self.norm(transformer_output[0])
                
                # 检查norm后的output
                check_nan(output, "output_after_norm", i)
                
                latents.append(output)
            
            # 在 stack 之前，添加原始模型的输出
            hidden_states = torch.stack([hidden_states] + latents, dim=1)
            
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            output = BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )
            return output if return_dict else output.to_tuple()

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
        # 替换原始的MistralModel为MistralMTPModel
        self.model = MistralMTPModel(config)
    
    # 添加这个方法来帮助 PEFT 找到正确的层
    def get_decoder(self):
        """Helper method to help PEFT locate the decoder"""
        return self.model

    def get_wrapped_policy(self):
        """返回 FSDP 包装策略"""
        from torch.distributed.fsdp.wrap import (
            size_based_auto_wrap_policy,
            transformer_auto_wrap_policy,
        )
        from functools import partial
        from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

        # 定义不需要包装的模块类型
        no_wrap_modules = {
            MistralMTPModel,  # 主模型
            nn.ModuleList,  # ModuleList
            nn.Linear,  # 线性层（包括 lm_head 和 future_lm_heads）
        }

        # 使用 transformer_auto_wrap_policy
        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={MistralDecoderLayer},  # Decoder层需要被分割
            no_wrap_modules=no_wrap_modules,  # 指定不需要包装的模块
        )

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
        if hasattr(model.model, "mtp_modules"):
            for module in model.model.mtp_modules:
                # 确保 projection 和 transformer 都是可训练的
                module['projection'].requires_grad_(True)
                module['transformer'].requires_grad_(True)
                logger.info_rank0(f"Set projection and transformer to trainable")
    
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