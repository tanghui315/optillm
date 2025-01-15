from typing import Optional, Union, Tuple, Unpack
import torch
import torch.nn as nn
from transformers.models.mistral.modeling_mistral import MistralModel, MistralForCausalLM, MistralDecoderLayer
from transformers.cache_utils import Cache,DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from torch.nn import CrossEntropyLoss

logger = logging.get_logger(__name__)
class MistralMTPModel(MistralModel):
    
    def __init__(self, config):
        # # 使用config中的torch_dtype，如果没有则默认使用bfloat16
        # config.torch_dtype = getattr(config, "torch_dtype", torch.bfloat16)
        super().__init__(config)
        self.n_future_tokens = getattr(config, "n_future_tokens", 1)  # 默认为1，兼容原始模型
        
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

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache()

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
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            
            # 通过主干网络
            for i, layer in enumerate(self.layers[:-self.n_future_tokens]):
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
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
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )
                hidden_states = layer_outputs[0]
            
            trunk_output = hidden_states
            
            # 通过预测头
            latents = []
            prev_latent = None  # 上一个预测头的输出
            for i in range(self.n_future_tokens):
                # 使用torch.cuda.empty_cache()清理不需要的缓存
                torch.cuda.empty_cache()
                
                if prev_latent is not None:
                    trunk_input = trunk_output + prev_latent  # 融合主干输出与上一个预测头的输出
                else:
                    trunk_input = trunk_output  # 初始情况下，仅使用主干输出
                head_layer = self.layers[-self.n_future_tokens + i]
                layer_outputs = head_layer(
                    trunk_input,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )
                prev_latent = self.norm(layer_outputs[0])
                latents.append(prev_latent)
                
                # 删除不需要的中间变量
                del layer_outputs
                
            # 堆叠多个预测结果
            hidden_states = torch.stack(latents, dim=-2)
            
            if not return_dict:
                return (hidden_states,)
            
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
                hidden_states=output_hidden_states,
                attentions=output_attentions,
            )

# 用于语言模型的MTP模型
class MistralMTPForCausalLM(MistralForCausalLM):
    _no_split_modules = ["MistralDecoderLayer"]
    _tied_weights_keys = ["lm_head.weight", "future_lm_heads.weight"]
    _tp_plan = {"lm_head": "colwise_rep", "future_lm_heads": "colwise_rep"}
    # # 添加这个类属性来帮助 PEFT 识别模型结构
    # base_model_prefix = "model"
    
    def __init__(self, config):
        super().__init__(config)
        # 替换原始的MistralModel为MistralMTPModel
        self.model = MistralMTPModel(config)
        
        # 为每个future token创建一个lm_head
        if getattr(config, "n_future_tokens", 1) > 1:
            self.future_lm_heads = nn.ModuleList([
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                for _ in range(config.n_future_tokens)
            ])
            for head in self.future_lm_heads:
                head.weight.data.copy_(self.lm_head.weight.data)
    
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
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
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
        
        outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs[0]  # [batch, n_future_tokens, seq_len, hidden_size]
        
        # MTP模式的logits计算
        logits = []
        for i in range(self.model.n_future_tokens):
            step_hidden = hidden_states[:, i]  # [batch, seq_len, hidden_size]
            step_logits = self.future_lm_heads[i](step_hidden)
            logits.append(step_logits)
        logits = torch.stack(logits, dim=1)  # [batch, n_future_tokens, seq_len, vocab_size]

        loss = None
        if labels is not None:
            # 转换为float32以避免精度问题
            logits_float = logits.float()
            
            # 准备多步预测的标签
            shifted_labels = labels[:, 1:]  # 移除第一个token
            future_labels = []
            for i in range(self.model.n_future_tokens):
                # 对每个预测步长,取对应位置的标签
                step_labels = shifted_labels[:, i:i+shifted_labels.size(1)-self.model.n_future_tokens+1]
                future_labels.append(step_labels)
            future_labels = torch.stack(future_labels, dim=1)  # [batch_size, n_future_tokens, seq_len]
            
            # 计算每个预测头的loss
            loss_fct = CrossEntropyLoss()
            losses = []
            for i in range(self.model.n_future_tokens):
                step_logits = logits_float[:, i]  # [batch_size, seq_len, vocab_size]
                step_labels = future_labels[:, i]  # [batch_size, seq_len]
                
                # 重塑logits以匹配CrossEntropyLoss期望的形状
                shift_logits = step_logits[..., :-1, :].contiguous()
                shift_labels = step_labels[..., 1:].contiguous()
                
                # 计算这一步的loss
                step_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                   shift_labels.view(-1))
                losses.append(step_loss)
            
            # 合并所有预测头的loss，使用加权平均
            weights = torch.tensor([1/(i+1) for i in range(self.model.n_future_tokens)], 
                                 device=losses[0].device)
            weights = weights / weights.sum()  # 归一化权重
            loss = torch.stack(losses) @ weights  # 加权求和

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        loaded_keys,
        resolved_archive_file,
        pretrained_model_name_or_path,
        ignore_mismatched_sizes=False,
        sharded_metadata=None,
        _fast_init=True,
        low_cpu_mem_usage=False,
        device_map=None,
        offload_folder=None,
        offload_state_dict=None,
        dtype=None,
        hf_quantizer=None,
        keep_in_fp32_modules=None,
        gguf_path=None,
        weights_only=True,
    ):
        """重写父类的 _load_pretrained_model 方法，确保参数完整"""
        
        # 处理新增的 future_lm_heads
        if getattr(model.config, "n_future_tokens", 1) > 1:
            # 确保 state_dict 存在
            if state_dict is None:
                state_dict = {}
                logger.warning_rank0("state_dict is None, initializing an empty one")
            
            for i in range(model.config.n_future_tokens):
                key = f"future_lm_heads.{i}.weight"
                # 只有当 key 不存在时才进行初始化
                if key not in state_dict:
                    if "lm_head.weight" in state_dict:
                        # 使用原始 lm_head 的权重初始化
                        logger.info_rank0(f"Initializing {key} with lm_head weights")
                        state_dict[key] = state_dict["lm_head.weight"].clone()
                    else:
                        logger.warning_rank0(f"Cannot find lm_head.weight to initialize {key}")
                    if loaded_keys is not None:  # 也要检查 loaded_keys
                        loaded_keys.append(key)
                else:
                    logger.info_rank0(f"Found existing weights for {key}")
        
        # 调用父类的加载方法
        return super()._load_pretrained_model(
            model=model,
            state_dict=state_dict,
            loaded_keys=loaded_keys,
            resolved_archive_file=resolved_archive_file,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            sharded_metadata=sharded_metadata,
            _fast_init=_fast_init,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            dtype=dtype,
            hf_quantizer=hf_quantizer,
            keep_in_fp32_modules=keep_in_fp32_modules,
            gguf_path=gguf_path,
            weights_only=weights_only,
        )