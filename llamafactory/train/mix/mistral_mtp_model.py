from typing import Optional, Union, Tuple, List
import torch
import torch.nn as nn
from transformers.models.mistral.modeling_mistral import MistralModel, MistralForCausalLM, MistralDecoderLayer
from transformers.cache_utils import Cache,DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from torch.nn import CrossEntropyLoss
from ...extras.logging import get_logger
import math
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
            # 创建独立的MTP模块，并确保每个transformer有唯一的layer_idx
            base_idx = len(self.layers)  # 基础模型的层数
            self.mtp_modules = nn.ModuleList([
                nn.ModuleDict({
                    'projection': nn.Linear(config.hidden_size * 2, config.hidden_size),
                    'proj_act': nn.GELU(),
                    'pre_transformer_norm': nn.LayerNorm(config.hidden_size),
                    'transformer': MistralDecoderLayer(config, base_idx + i)
                }) for i in range(self.n_future_tokens)
            ])
            
            # 初始化MTP模块
            for mtp_module in self.mtp_modules:
                # 投影层使用 xavier 初始化
                nn.init.xavier_uniform_(mtp_module['projection'].weight)
                if mtp_module['projection'].bias is not None:
                    nn.init.zeros_(mtp_module['projection'].bias)
                
                # LayerNorm 层保持默认初始化
                
                # Transformer 层的特定初始化
                transformer = mtp_module['transformer']
                
                # 注意力层初始化
                nn.init.xavier_uniform_(transformer.self_attn.q_proj.weight, gain=1/math.sqrt(2))
                nn.init.xavier_uniform_(transformer.self_attn.k_proj.weight, gain=1/math.sqrt(2))
                nn.init.xavier_uniform_(transformer.self_attn.v_proj.weight, gain=1/math.sqrt(2))
                nn.init.xavier_uniform_(transformer.self_attn.o_proj.weight)
                if hasattr(transformer.self_attn.o_proj, 'bias') and transformer.self_attn.o_proj.bias is not None:
                    nn.init.zeros_(transformer.self_attn.o_proj.bias)
                
                # FFN 层初始化
                nn.init.xavier_uniform_(transformer.mlp.gate_proj.weight)
                nn.init.xavier_uniform_(transformer.mlp.up_proj.weight)
                nn.init.xavier_uniform_(transformer.mlp.down_proj.weight)
                
                # RMSNorm 层保持默认初始化
                # transformer.input_layernorm 和 transformer.post_attention_layernorm
        
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
                        past_key_value=None if not use_cache else past_key_values,
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
                if past_key_values is not None and i in past_key_values.key_cache:
                    key = past_key_values.key_cache[i]
                    value = past_key_values.value_cache[i]
                    print(f"  - updated key shape: {key.shape if key is not None else None}")
                    print(f"  - updated value shape: {value.shape if value is not None else None}")
            
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
                projected = mtp_module['proj_act'](projected)  # 添加激活函数
                projected = mtp_module['pre_transformer_norm'](projected)  # 添加归一化
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
                if past_key_values is not None:
                    layer_idx = len(self.layers) + i
                    if layer_idx in past_key_values.key_cache:
                        key = past_key_values.key_cache[layer_idx]
                        value = past_key_values.value_cache[layer_idx]
                        print(f"  - key shape: {key.shape if key is not None else None}")
                        print(f"  - value shape: {value.shape if value is not None else None}")

                transformer_output = mtp_module['transformer'](
                    projected,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=None if not use_cache else past_key_values,  # 当 use_cache=False 时不使用缓存
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
                    if layer_idx in past_key_values.key_cache:
                        key = past_key_values.key_cache[layer_idx]
                        value = past_key_values.value_cache[layer_idx]
                        print(f"  - updated key shape: {key.shape if key is not None else None}")
                        print(f"  - updated value shape: {value.shape if value is not None else None}")
                
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
            logits_float = logits.to(dtype=self.config.torch_dtype)
            loss_fct = CrossEntropyLoss()
            losses = []
            
            # 计算所有模块的 loss（包括原始模型）
            for i in range(self.model.n_future_tokens + 1):
                step_logits = logits_float[:, i]
                
                # 检查step_logits是否有NaN
                if torch.isnan(step_logits).any():
                    print(f"\n[Warning] Found NaN in step_logits before masking for step {i}")
                    print(f"- step_logits shape: {step_logits.shape}")
                    print(f"- step_logits stats before masking:")
                    print(f"  * min={step_logits.min().item() if not torch.isnan(step_logits.min()) else 'nan'}")
                    print(f"  * max={step_logits.max().item() if not torch.isnan(step_logits.max()) else 'nan'}")
                    print(f"  * mean={step_logits.mean().item() if not torch.isnan(step_logits.mean()) else 'nan'}")
                    print(f"  * % of NaN: {torch.isnan(step_logits).float().mean().item() * 100:.2f}%")
                
                target_tokens = labels.clone()
                
                print(f"\n[Debug] Loss computation step {i}:")
                print(f"- step_logits stats: min={step_logits.min().item():.4f}, max={step_logits.max().item():.4f}, mean={step_logits.mean().item():.4f}")
                print(f"- target_tokens unique values: {torch.unique(target_tokens).tolist()}")
                
                # 对于每个模块，预测不同的未来位置
                if i == 0:
                    # Main Module: 预测t+1
                    target_tokens[:, 0] = -100  # 不预测第一个token
                else:
                    # MTP模块: MTP1预测t+2, MTP2预测t+3
                    # 将前i+1个位置设为-100，这样:
                    # MTP1 (i=1): 从位置2开始预测
                    # MTP2 (i=2): 从位置3开始预测
                    target_tokens[:, :i+1] = -100
                
                print(f"After masking:")
                valid_targets = torch.sum(target_tokens != -100)
                print(f"- valid targets (not -100): {valid_targets.item()}")
                print(f"- target_tokens sample: {target_tokens[0, :10].tolist()}")
                
                reshaped_logits = step_logits.reshape(-1, step_logits.size(-1))
                reshaped_targets = target_tokens.reshape(-1)
                
                # 检查logits是否有异常值
                print(f"Reshaped logits stats:")
                print(f"- min={reshaped_logits.min().item():.4f}, max={reshaped_logits.max().item():.4f}")
                print(f"- mean={reshaped_logits.mean().item():.4f}, std={reshaped_logits.std().item():.4f}")
                print(f"- has_nan: {torch.isnan(reshaped_logits).any().item()}")
                print(f"- has_inf: {torch.isinf(reshaped_logits).any().item()}")
                
                # 使用梯度裁剪来防止梯度爆炸
                step_logits = torch.clamp(step_logits, min=-100, max=100)
                
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