from typing import Optional, Union, Tuple, Unpack
import torch
import torch.nn as nn
from transformers.models.mistral.modeling_mistral import MistralModel
from transformers.cache_utils import Cache,DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)
class MistralMTPModel(MistralModel):
    
    def __init__(self, config):
        # 使用config中的torch_dtype，如果没有则默认使用bfloat16
        config.torch_dtype = getattr(config, "torch_dtype", torch.bfloat16)
        super().__init__(config)
        self.n_future_tokens = getattr(config, "n_future_tokens", 1)  # 默认为1，兼容原始模型
        
        # 只有在n_future_tokens > 1时才分离主干网络和预测头
        if self.n_future_tokens > 1:
            # 使用主干网络的最后几层作为预测头
            self.main_layers = nn.ModuleList(self.layers[:-1])  # 除最后一层外的所有层
            self.shared_layer = self.layers[-1]  # 最后一层作为共享层
            
            # 为每个future token创建一个轻量级的预测头
            self.prediction_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.GELU(),
                    nn.Linear(config.hidden_size, config.hidden_size)
                ) for _ in range(self.n_future_tokens)
            ])
        
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
            
            # 通过共享层
            shared_output = self.shared_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )[0]
            
            # 通过预测头
            latents = []
            prev_latent = None
            for head in self.prediction_heads:
                if prev_latent is not None:
                    trunk_input = shared_output + prev_latent
                else:
                    trunk_input = shared_output
                
                # 使用轻量级预测头
                prev_latent = head(trunk_input)
                latents.append(prev_latent)
                
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