from typing import Optional, Union, Tuple, Unpack
import torch
import torch.nn as nn
from transformers.models.mistral.modeling_mistral import MistralModel, MistralForCausalLM
from transformers.cache_utils import Cache,DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from torch.nn import CrossEntropyLoss

logger = logging.get_logger(__name__)
class MistralMTPModel(MistralModel):
    
    def __init__(self, config):
        # 使用config中的torch_dtype，如果没有则默认使用bfloat16
        config.torch_dtype = getattr(config, "torch_dtype", torch.bfloat16)
        super().__init__(config)
        self.n_future_tokens = getattr(config, "n_future_tokens", 1)  # 默认为1，兼容原始模型
        
        # 只有在n_future_tokens > 1时才分离主干网络和预测头
        if self.n_future_tokens > 1:
            main_layers = self.layers[:-self.n_future_tokens]  # 保留主干层
            head_layers = self.layers[-self.n_future_tokens:]  # 取最后n_future_tokens层作为预测头
            
            self.main_layers = nn.ModuleList(main_layers)
            self.prediction_heads = nn.ModuleList(head_layers)
        
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
        with torch.autocast("cuda", dtype=self.config.torch_dtype):
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
            for layer in self.main_layers:
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
            for head in self.prediction_heads:
                # 使用torch.cuda.empty_cache()清理不需要的缓存
                torch.cuda.empty_cache()
                
                if prev_latent is not None:
                    trunk_input = trunk_output + prev_latent  # 融合主干输出与上一个预测头的输出
                else:
                    trunk_input = trunk_output  # 初始情况下，仅使用主干输出
                layer_outputs = head(
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
    # 添加wrap_policy标记
    _keys_to_ignore_on_load_missing = [r"model.layers.\d+.self_attn.rotary_emb.inv_freq"]
    _no_split_modules = ["MistralDecoderLayer"]
    
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
            # 用原始lm_head的权重初始化
            for head in self.future_lm_heads:
                head.weight.data.copy_(self.lm_head.weight.data)

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